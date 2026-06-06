from __future__ import annotations

from pathlib import Path


def hubert_npz_paths(slice_zip: Path, data_dir: Path) -> tuple[Path, Path]:
    stem = slice_zip.stem
    return (
        data_dir / f"{stem}_hubert_in.npz",
        data_dir / f"{stem}_hubert_out.npz",
    )
