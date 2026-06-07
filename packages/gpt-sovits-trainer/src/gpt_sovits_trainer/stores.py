"""Lazy loaders for orchestrator NPZ/ZIP modality stores."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ModalityStores:
    hubert_out_npz: Path
    sv_out_npz: Path
    semantic_out_npz: Path
    wav32k_zip: Path

    def __post_init__(self) -> None:
        for path in (self.hubert_out_npz, self.sv_out_npz, self.semantic_out_npz, self.wav32k_zip):
            if not path.is_file():
                raise FileNotFoundError(f"Missing modality artifact: {path}")

        self._hubert = np.load(self.hubert_out_npz, allow_pickle=False)
        self._sv = np.load(self.sv_out_npz, allow_pickle=False)
        self._semantic = np.load(self.semantic_out_npz, allow_pickle=False)
        self._wav32k = zipfile.ZipFile(self.wav32k_zip, "r")
        self._wav32k_sizes = {
            Path(name).name: self._wav32k.getinfo(name).file_size
            for name in self._wav32k.namelist()
            if name.lower().endswith(".wav") and not name.endswith("/")
        }

    def close(self) -> None:
        self._hubert.close()
        self._sv.close()
        self._semantic.close()
        self._wav32k.close()

    def hubert(self, filename: str) -> np.ndarray:
        return self._hubert[filename]

    def sv(self, filename: str) -> np.ndarray:
        return self._sv[filename]

    def semantic(self, filename: str) -> np.ndarray:
        return self._semantic[filename]

    def wav32k_bytes(self, filename: str) -> bytes:
        return self._wav32k.read(filename)

    def wav32k_size(self, filename: str) -> int:
        return self._wav32k_sizes[filename]

    def keys(self) -> set[str]:
        return (
            set(self._hubert.files)
            & set(self._sv.files)
            & set(self._semantic.files)
            & set(self._wav32k_sizes)
        )
