"""Assemble-manifest backed s2 dataset (v2Pro)."""

from __future__ import annotations

import csv
import io
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm

from gpt_sovits_trainer.phoneme import phones_to_ids
from gpt_sovits_trainer.stores import ModalityStores


class AssembleTextAudioSpeakerLoader(torch.utils.data.Dataset):
    """Drop-in replacement for module.data_utils.TextAudioSpeakerLoader (v2Pro)."""

    def __init__(self, hparams, stores: ModalityStores, assemble_csv: str, version=None, val=False):
        from module.mel_processing import spectrogram_torch

        self._spectrogram_torch = spectrogram_torch
        self.stores = stores
        self.is_v2Pro = version in {"v2Pro", "v2ProPlus"}
        if not self.is_v2Pro:
            raise ValueError("AssembleTextAudioSpeakerLoader only supports v2Pro/v2ProPlus")

        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.val = val
        version_env = os.environ.get("version")

        rows = [
            row
            for row in csv.DictReader(io.open(assemble_csv, encoding="utf-8-sig"))
            if row.get("eligible_sovits", "").lower() == "true" and row.get("status") == "ok"
        ]
        filenames = sorted({row["filename"] for row in rows} & stores.keys())
        row_by_name = {row["filename"]: row for row in rows if row["filename"] in filenames}

        tmp = sorted(filenames)
        leng = len(tmp)
        min_num = 100
        if leng < min_num:
            expanded: list[str] = []
            for _ in range(max(2, int(min_num / leng))):
                expanded += tmp
            tmp = expanded

        random.seed(1234)
        random.shuffle(tmp)

        audiopaths_sid_text_new = []
        lengths = []
        skipped_phone = 0
        skipped_dur = 0
        for audiopath in tqdm(tmp):
            try:
                phoneme = row_by_name[audiopath]["phones"].split(" ")
                phoneme_ids = phones_to_ids(phoneme, version=version_env)
            except Exception:
                skipped_phone += 1
                continue

            size = stores.wav32k_size(audiopath)
            duration = size / self.sampling_rate / 2
            if duration == 0:
                skipped_dur += 1
                continue
            if 54 > duration > 0.6 or self.val:
                audiopaths_sid_text_new.append([audiopath, phoneme_ids])
                lengths.append(size // (2 * self.hop_length))
            else:
                skipped_dur += 1

        print("skipped_phone:", skipped_phone, ", skipped_dur:", skipped_dur)
        print("total left:", len(audiopaths_sid_text_new))
        if len(audiopaths_sid_text_new) <= 1:
            raise RuntimeError("Not enough s2 training samples after filtering")
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def _load_wav_norm(self, filename: str) -> torch.Tensor:
        import soundfile as sf

        wav_bytes = self.stores.wav32k_bytes(filename)
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sampling_rate:
            raise ValueError(f"Unexpected sample rate {sr} for {filename}, expected {self.sampling_rate}")
        audio = torch.FloatTensor(audio)
        if audio.abs().max() > 1.5:
            audio = audio / self.max_wav_value
        return audio.unsqueeze(0)

    def get_audio(self, filename: str):
        audio_norm = self._load_wav_norm(filename)
        spec = self._spectrogram_torch(
            audio_norm,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            center=False,
        )
        spec = torch.squeeze(spec, 0)
        return spec, audio_norm

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        audiopath, phoneme_ids = audiopath_sid_text
        text = torch.FloatTensor(phoneme_ids)
        spec, wav = self.get_audio(audiopath)
        ssl_np = self.stores.hubert(audiopath)
        ssl = torch.from_numpy(np.asarray(ssl_np)).float()
        if ssl.ndim == 3 and ssl.shape[1] != 768:
            ssl = ssl.transpose(1, 2)
        if ssl.shape[-1] != spec.shape[-1]:
            ssl = F.pad(ssl.float(), (0, 1), mode="replicate").to(ssl.dtype)
        ssl.requires_grad = False
        sv_emb = torch.from_numpy(np.asarray(self.stores.sv(audiopath))).float()
        return ssl, spec, wav, text, sv_emb

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)
