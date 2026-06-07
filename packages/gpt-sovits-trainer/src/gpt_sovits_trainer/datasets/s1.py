"""Assemble-manifest backed s1 dataset."""

from __future__ import annotations

import csv
import io
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from gpt_sovits_trainer.phoneme import phones_to_ids
from gpt_sovits_trainer.stores import ModalityStores


def batch_sequences(sequences: List[np.ndarray], axis: int = 0, pad_value: int = 0) -> np.ndarray:
    seq = sequences[0]
    ndim = seq.ndim
    if axis < 0:
        axis += ndim
    dtype = seq.dtype
    pad_value = dtype.type(pad_value)
    seq_lengths = [seq.shape[axis] for seq in sequences]
    max_length = int(np.max(seq_lengths))

    padded_sequences = []
    for seq, length in zip(sequences, seq_lengths):
        padding = [(0, 0)] * axis + [(0, max_length - length)] + [(0, 0)] * (ndim - axis - 1)
        padded_seq = np.pad(seq, padding, mode="constant", constant_values=pad_value)
        padded_sequences.append(padded_seq)
    return np.stack(padded_sequences)


class AssembleText2SemanticDataset(Dataset):
    """Drop-in replacement for AR.data.dataset.Text2SemanticDataset."""

    def __init__(
        self,
        assemble_csv: str,
        stores: ModalityStores,
        *,
        max_sec: int = 54,
        pad_val: int = 1024,
        min_ps_ratio: int = 3,
        max_ps_ratio: int = 25,
    ) -> None:
        self._batch_sequences = batch_sequences
        self.PAD = pad_val
        self.hz = int(os.environ.get("hz", "25hz")[:-2])
        self.max_sec = max_sec
        self.min_ps_ratio = min_ps_ratio
        self.max_ps_ratio = max_ps_ratio
        self.stores = stores
        version = os.environ.get("version")

        rows = list(csv.DictReader(io.open(assemble_csv, encoding="utf-8-sig")))
        self.semantic_phoneme: list[tuple[list[int], list[int]]] = []
        self.item_names: list[str] = []

        num_deleted_bigger = 0
        num_deleted_ps = 0
        for row in rows:
            if row.get("eligible_gpt", "").lower() != "true":
                continue
            if row.get("status") != "ok":
                continue
            filename = row["filename"]
            semantic_ids = stores.semantic(filename).astype(np.int32).tolist()
            if len(semantic_ids) > self.max_sec * self.hz:
                num_deleted_bigger += 1
                continue
            phones = row["phones"].split(" ")
            try:
                phoneme_ids = phones_to_ids(phones, version=version)
            except Exception:
                continue
            if len(phoneme_ids) > self.max_sec * self.hz / 2.5:
                num_deleted_ps += 1
                continue
            ps_ratio = len(phoneme_ids) / (len(semantic_ids) / self.hz)
            if ps_ratio > self.max_ps_ratio or ps_ratio < self.min_ps_ratio:
                num_deleted_ps += 1
                continue
            self.semantic_phoneme.append((semantic_ids, phoneme_ids))
            self.item_names.append(filename)

        min_num = 100
        length = len(self.semantic_phoneme)
        if length < min_num:
            tmp_pairs = self.semantic_phoneme
            tmp_names = self.item_names
            self.semantic_phoneme = []
            self.item_names = []
            for _ in range(max(2, int(min_num / length))):
                self.semantic_phoneme += tmp_pairs
                self.item_names += tmp_names

        if num_deleted_bigger:
            print(f"deleted {num_deleted_bigger} samples longer than {self.max_sec}s")
        if num_deleted_ps:
            print(
                f"deleted {num_deleted_ps} samples outside phoneme/sec "
                f"[{self.min_ps_ratio}, {self.max_ps_ratio}]"
            )
        print("dataset.__len__():", self.__len__())

    def __len__(self) -> int:
        return len(self.semantic_phoneme)

    def __getitem__(self, idx: int) -> Dict:
        semantic_ids, phoneme_ids = self.semantic_phoneme[idx]
        return {
            "idx": idx,
            "phoneme_ids": phoneme_ids,
            "phoneme_ids_len": len(phoneme_ids),
            "semantic_ids": semantic_ids,
            "semantic_ids_len": len(semantic_ids),
            "bert_feature": None,
        }

    def get_sample_length(self, idx: int) -> float:
        semantic_ids = self.semantic_phoneme[idx][0]
        return 1.0 * len(semantic_ids) / self.hz

    def collate(self, examples: List[Dict]) -> Dict:
        sample_index: List[int] = []
        phoneme_ids: List[np.ndarray] = []
        phoneme_ids_lens: List[int] = []
        semantic_ids: List[np.ndarray] = []
        semantic_ids_lens: List[int] = []

        for item in examples:
            sample_index.append(item["idx"])
            phoneme_ids.append(np.array(item["phoneme_ids"], dtype=np.int64))
            semantic_ids.append(np.array(item["semantic_ids"], dtype=np.int64))
            phoneme_ids_lens.append(item["phoneme_ids_len"])
            semantic_ids_lens.append(item["semantic_ids_len"])

        phoneme_ids_arr = self._batch_sequences(phoneme_ids)
        semantic_ids_arr = self._batch_sequences(semantic_ids, pad_value=self.PAD)

        phoneme_ids_t = torch.tensor(phoneme_ids_arr)
        semantic_ids_t = torch.tensor(semantic_ids_arr)
        phoneme_ids_lens_t = torch.tensor(phoneme_ids_lens)
        semantic_ids_lens_t = torch.tensor(semantic_ids_lens)
        bert_padded = torch.FloatTensor(len(examples), 1024, max(phoneme_ids_lens))
        bert_padded.zero_()

        return {
            "ids": sample_index,
            "phoneme_ids": phoneme_ids_t,
            "phoneme_ids_len": phoneme_ids_lens_t,
            "semantic_ids": semantic_ids_t,
            "semantic_ids_len": semantic_ids_lens_t,
            "bert_feature": bert_padded,
        }
