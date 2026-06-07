from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np


def load_npz_keys(npz_path: Path) -> set[str]:
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    return set(np.load(npz_path).files)


def load_wav32k_entry_sizes(zip_path: Path) -> dict[str, int]:
    if not zip_path.is_file():
        raise FileNotFoundError(f"wav32k ZIP not found: {zip_path}")

    sizes: dict[str, int] = {}
    with zipfile.ZipFile(zip_path, "r") as archive:
        for name in archive.namelist():
            if name.lower().endswith(".wav") and not name.endswith("/"):
                sizes[Path(name).name] = archive.getinfo(name).file_size
    if not sizes:
        raise ValueError(f"No WAV entries found in wav32k ZIP: {zip_path}")
    return sizes


def modality_audit(
    filename: str,
    *,
    hubert_out: np.lib.npyio.NpzFile,
    semantic_out: np.lib.npyio.NpzFile,
    sv_out: np.lib.npyio.NpzFile,
    wav32k_sizes: dict[str, int],
) -> dict[str, int]:
    hubert = hubert_out[filename]
    semantic = semantic_out[filename]
    sv = sv_out[filename]
    if filename not in wav32k_sizes:
        raise KeyError(f"Missing wav32k entry: {filename}")

    return {
        "semantic_count": int(semantic.shape[0]),
        "hubert_T": int(hubert.shape[1]),
        "sv_dim": int(sv.shape[-1]),
        "wav32k_bytes": int(wav32k_sizes[filename]),
    }
