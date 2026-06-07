"""Native PyTorch training loop for GPT s1, explicitly defining forward/backward passes."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gpt_sovits_trainer.paths import DATA_10_DIR, EXP_NAME, PRETRAINED_S1, ensure_data_dirs
from gpt_sovits_trainer.stores import ModalityStores
from gpt_sovits_trainer.vendor_env import bootstrap_vendor

def train_s1(
    assemble_csv: Path,
    stores: ModalityStores,
    *,
    epochs: int = 2,
    batch_size: int = 4,
) -> Path:
    bootstrap_vendor()
    ensure_data_dirs()

    # 1. Configuration matching vendor defaults
    config = {
        "train": {
            "batch_size": batch_size,
            "epochs": epochs,
            "if_dpo": False,
        },
        "optimizer": {
            "lr": 0.01,
            "lr_init": 0.00001,
            "lr_end": 0.0001,
            "warmup_steps": 2000,
            "decay_steps": 40000,
        },
        "data": {
            "max_sec": 54,
            "pad_val": 1024,
        },
        "model": {
            "vocab_size": 1025,
            "phoneme_vocab_size": 732,
            "embedding_dim": 512,
            "hidden_dim": 512,
            "head": 16,
            "linear_units": 2048,
            "n_layer": 24,
            "dropout": 0,
            "EOS": 1024,
            "random_bert": 0,
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"S1 Training on {device}")

    # 2. Dataset & DataLoader (using our existing Assemble wrapper)
    from gpt_sovits_trainer.datasets.s1 import AssembleText2SemanticDataset
    dataset = AssembleText2SemanticDataset(
        assemble_csv=str(assemble_csv),
        stores=stores,
        max_sec=54,
        pad_val=1024,
    )
    
    from AR.data.bucket_sampler import DistributedBucketSampler
    sampler = DistributedBucketSampler(dataset, batch_size=batch_size, num_replicas=1, rank=0)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=dataset.collate,
        num_workers=0, # Keep single process for stability on Windows
    )

    # 3. Model Initialization
    from AR.models.t2s_model import Text2SemanticDecoder
    model = Text2SemanticDecoder(config=config, top_k=3)
    
    if PRETRAINED_S1.is_file():
        print(f"Loading pretrained s1 weights from {PRETRAINED_S1}")
        pretrained_dict = torch.load(PRETRAINED_S1, map_location="cpu")
        state_dict = pretrained_dict["weight"] if "weight" in pretrained_dict else pretrained_dict
        
        # Remove 'model.' prefix typically found in Lightning checkpoints
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k[6:] if k.startswith("model.") else k
            new_state_dict[new_k] = v
            
        model.load_state_dict(new_state_dict)
        
    model = model.to(device)
    model.train()

    # 4. Optimizer & Scheduler
    from AR.modules.optim import ScaledAdam
    from AR.modules.lr_schedulers import WarmupCosineLRSchedule
    
    model_parameters = model.parameters()
    parameters_names = [[name for name, _ in model.named_parameters()]]
    
    optimizer = ScaledAdam(
        model_parameters,
        lr=config["optimizer"]["lr"],
        betas=(0.9, 0.95),
        clipping_scale=2.0,
        parameters_names=parameters_names,
        show_dominant_parameters=False,
        clipping_update_period=1000,
    )
    
    scheduler = WarmupCosineLRSchedule(
        optimizer,
        init_lr=config["optimizer"]["lr_init"],
        peak_lr=config["optimizer"]["lr"],
        end_lr=config["optimizer"]["lr_end"],
        warmup_steps=config["optimizer"]["warmup_steps"],
        total_steps=config["optimizer"]["decay_steps"],
    )

    # 5. Core Training Loop
    global_step = 0
    for epoch in range(1, epochs + 1):
        pbar = tqdm(dataloader, desc=f"S1 Epoch {epoch}/{epochs}")
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(pbar):
            phoneme_ids = batch["phoneme_ids"].to(device)
            phoneme_ids_len = batch["phoneme_ids_len"].to(device)
            semantic_ids = batch["semantic_ids"].to(device)
            semantic_ids_len = batch["semantic_ids_len"].to(device)
            bert_feature = batch["bert_feature"].to(device)
            
            # Forward pass: Text2SemanticDecoder inherently returns (loss, accuracy) 
            # by comparing shifted outputs against semantic targets
            loss, acc = model(
                phoneme_ids, 
                phoneme_ids_len, 
                semantic_ids, 
                semantic_ids_len, 
                bert_feature
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping & Optimizer step (accumulate gradients every 4 steps to simulate larger batch if needed, here we step every time for simplicity but match author's batch size scale)
            if (batch_idx + 1) % 4 == 0 or (batch_idx + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
            global_step += 1

        print(f"Epoch {epoch} finished. Avg Loss: {total_loss / len(dataloader):.4f}")

    # 6. Save Artifacts
    # In S1, the saved weight is typically stripped of optimizers to save space
    out_path = DATA_10_DIR / f"{EXP_NAME}.ckpt"
    save_dict = {
        # api.py loads into Text2SemanticLightningModule, keys must be model.*
        "weight": {
            (k if k.startswith("model.") else f"model.{k}"): v.half().cpu()
            for k, v in model.state_dict().items()
        },
        "config": config,
        "info": f"GPT-e{epochs}",
    }
    torch.save(save_dict, out_path)
    print(f"Saved S1 weights to {out_path}")
    return out_path
