from __future__ import annotations

from pathlib import Path


def wav32k_zip_path(slice_zip: Path, data_dir: Path) -> Path:
    return data_dir / f"{slice_zip.stem}_wav32k.zip"
