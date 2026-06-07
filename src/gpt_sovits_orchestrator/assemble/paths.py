from __future__ import annotations

from pathlib import Path


def assemble_manifest_path(g2p_csv_path: Path, output_dir: Path) -> Path:
    base_name = g2p_csv_path.stem.removesuffix("_manifest_g2p")
    return output_dir / f"{base_name}_manifest_assemble.csv"
