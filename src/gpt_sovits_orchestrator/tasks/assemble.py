from __future__ import annotations

from pathlib import Path

from prefect import task

from gpt_sovits_orchestrator.assemble.manifest import build_assemble_manifest
from gpt_sovits_orchestrator.config import (
    DATA_05_DIR,
    DATA_06_DIR,
    DATA_07_DIR,
    DATA_08_DIR,
)


@task(name="assemble-manifest", log_prints=True)
def assemble_manifest(
    zip_path: Path,
    g2p_csv_path: Path,
    output_dir: Path,
    *,
    hubert_data_dir: Path = DATA_05_DIR,
    sv_data_dir: Path = DATA_06_DIR,
    semantic_data_dir: Path = DATA_07_DIR,
    wav32k_data_dir: Path = DATA_08_DIR,
) -> Path:
    """Build assemble manifest CSV indexing NPZ/ZIP modalities by filename."""
    output_path, count = build_assemble_manifest(
        zip_path,
        g2p_csv_path,
        output_dir,
        hubert_data_dir=hubert_data_dir,
        sv_data_dir=sv_data_dir,
        semantic_data_dir=semantic_data_dir,
        wav32k_data_dir=wav32k_data_dir,
    )
    print(f"Saved assemble manifest: {output_path} ({count} rows)")
    return output_path
