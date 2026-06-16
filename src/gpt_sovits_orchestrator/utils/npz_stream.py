from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np


class NpzStreamWriter:
    """Write a NumPy .npz archive one array at a time without holding all arrays in RAM."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._archive = zipfile.ZipFile(
            self.path,
            "w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        )
        self.count = 0

    def add(self, key: str, array: np.ndarray) -> None:
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        self._archive.writestr(f"{key}.npy", buffer.getvalue())
        self.count += 1

    def close(self) -> None:
        self._archive.close()

    def __enter__(self) -> NpzStreamWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
