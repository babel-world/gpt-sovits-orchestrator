from __future__ import annotations

from pathlib import Path


def semantic_npz_paths(hubert_out_npz: Path, data_dir: Path) -> tuple[Path, Path]:
    stem = hubert_out_npz.stem.removesuffix("_hubert_out")
    return (
        data_dir / f"{stem}_semantic_in.npz",
        data_dir / f"{stem}_semantic_out.npz",
    )
