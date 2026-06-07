from __future__ import annotations

import numpy as np


def hubert_to_semantic_input(hubert_arr: np.ndarray) -> np.ndarray:
    """(1, T, 768) float32 -> (1, 768, T) float32."""
    arr = np.asarray(hubert_arr, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 1 or arr.shape[2] != 768:
        raise ValueError(f"Expected hubert shape (1, T, 768), got {arr.shape}")
    return np.transpose(arr, (0, 2, 1)).astype(np.float32, copy=False)
