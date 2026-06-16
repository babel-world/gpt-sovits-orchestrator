"""Native PyTorch training loop for SoVITS s2 GAN, explicitly defining generator/discriminator logic."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from gpt_sovits_trainer.paths import GPT_SOVITS_ROOT, PRETRAINED_S2D, PRETRAINED_S2G, ensure_data_dirs
from gpt_sovits_trainer.stores import ModalityStores
from gpt_sovits_trainer.vendor_env import bootstrap_vendor

def train_s2(
    assemble_csv: Path,
    stores: ModalityStores,
    *,
    epochs: int = 2,
    batch_size: int = 4,
    exp_name: str = "manbo",
    weights_dir: Path | None = None,
) -> Path:
    bootstrap_vendor()
    layout = ensure_data_dirs()
    output_dir = weights_dir or layout.weights_dir

    # 1. Configuration matching vendor defaults
    with open(GPT_SOVITS_ROOT / "configs" / "s2v2Pro.json", encoding="utf-8") as handle:
        config = json.load(handle)
        
    class Dict2Obj:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, Dict2Obj(v) if isinstance(v, dict) else v)
    hps = Dict2Obj(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"S2 Training on {device}")

    # 2. Dataset & DataLoader
    from gpt_sovits_trainer.datasets.s2 import AssembleTextAudioSpeakerLoader
    from gpt_sovits_trainer.datasets.s2_utils import DistributedBucketSampler, TextAudioSpeakerCollate
    
    dataset = AssembleTextAudioSpeakerLoader(
        hps.data,
        stores=stores,
        assemble_csv=str(assemble_csv),
        version="v2Pro"
    )
    
    sampler = DistributedBucketSampler(
        dataset,
        batch_size=batch_size,
        boundaries=[32, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        num_replicas=1,
        rank=0,
        shuffle=True,
    )
    collate_fn = TextAudioSpeakerCollate(version="v2Pro")
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # 3. Model Initialization
    from module.models import SynthesizerTrn, MultiPeriodDiscriminator

    model_cfg = {**config["model"], "version": "v2Pro"}
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **model_cfg,
    ).to(device)
    
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm, version="v2Pro").to(device)
    
    if PRETRAINED_S2G.is_file():
        print(f"Loading pretrained s2 Generator from {PRETRAINED_S2G}")
        net_g.load_state_dict(torch.load(PRETRAINED_S2G, map_location="cpu", weights_only=False)["weight"], strict=False)
    if PRETRAINED_S2D.is_file():
        print(f"Loading pretrained s2 Discriminator from {PRETRAINED_S2D}")
        net_d.load_state_dict(torch.load(PRETRAINED_S2D, map_location="cpu", weights_only=False)["weight"], strict=False)
        
    net_g.train()
    net_d.train()

    # 4. Optimizers
    learning_rate = hps.train.learning_rate
    optim_g = torch.optim.AdamW(
        [
            {"params": filter(lambda p: p.requires_grad, net_g.parameters()), "lr": learning_rate},
        ],
        learning_rate, betas=hps.train.betas, eps=hps.train.eps
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        learning_rate, betas=hps.train.betas, eps=hps.train.eps
    )

    # 5. Losses and Processing tools
    from module.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
    from module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
    from module.commons import slice_segments

    # 6. Core Training Loop (GAN alternating loop)
    global_step = 0
    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch)
        pbar = tqdm(dataloader, desc=f"S2 Epoch {epoch}/{epochs}")
        total_loss_g, total_loss_d = 0.0, 0.0

        for batch_idx, batch in enumerate(pbar):
            ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths, sv_emb = batch
            ssl = ssl.to(device)
            ssl.requires_grad = False
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device)
            text, text_lengths = text.to(device), text_lengths.to(device)
            sv_emb = sv_emb.to(device)

            y_hat, _, ids_slice, _, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), _ = net_g(
                ssl, spec, spec_lengths, text, text_lengths, sv_emb
            )

            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y = slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

            # --- B. Discriminator Step ---
            # Forward Discriminator with both real and generated audio
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            
            optim_d.zero_grad()
            loss_disc.backward()
            torch.nn.utils.clip_grad_norm_(net_d.parameters(), hps.train.grad_clip) if hasattr(hps.train, "grad_clip") else None
            optim_d.step()
            total_loss_d += loss_disc.item()

            # --- C. Generator Step ---
            # Forward Discriminator again but allowing gradients to flow to generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
            
            optim_g.zero_grad()
            loss_gen_all.backward()
            torch.nn.utils.clip_grad_norm_(net_g.parameters(), hps.train.grad_clip) if hasattr(hps.train, "grad_clip") else None
            optim_g.step()
            total_loss_g += loss_gen_all.item()

            pbar.set_postfix({"L_D": f"{loss_disc.item():.3f}", "L_G": f"{loss_gen_all.item():.3f}", "L_Mel": f"{loss_mel.item():.3f}"})
            global_step += 1

        print(f"Epoch {epoch} finished. Avg Gen Loss: {total_loss_g / len(dataloader):.4f}, Avg Disc Loss: {total_loss_d / len(dataloader):.4f}")

    # 7. Save Artifacts (vendor-compatible: config + v2Pro header)
    from io import BytesIO

    out_path = output_dir / f"{exp_name}.pth"
    save_dict = {
        # process_ckpt.savee skips enc_q; inference loads with strict=False after del enc_q
        "weight": {
            k: v.half().cpu()
            for k, v in net_g.state_dict().items()
            if "enc_q" not in k
        },
        "config": config,
        "info": f"{epochs}epoch_{global_step}iteration",
    }
    bio = BytesIO()
    torch.save(save_dict, bio)
    payload = bio.getvalue()
    out_path.write_bytes(b"05" + payload[2:])  # v2Pro magic header for api.py
    print(f"Saved S2 Generator weights to {out_path}")
    return out_path
