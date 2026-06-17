from __future__ import annotations

from pathlib import Path

from gpt_sovits_orchestrator.pipeline_dirs import PipelineStageDirs
from gpt_sovits_orchestrator.tasks.assemble import assemble_manifest
from gpt_sovits_orchestrator.tasks.g2p import g2p_manifest
from gpt_sovits_orchestrator.tasks.hubert import hubert_from_zip
from gpt_sovits_orchestrator.tasks.semantic import semantic_from_hubert_out
from gpt_sovits_orchestrator.tasks.sv import sv_from_zip
from gpt_sovits_orchestrator.tasks.wav32k import wav32k_from_zip


def orchestrator_tail(
    zip_path: Path,
    manifest_csv_path: Path,
    *,
    dirs: PipelineStageDirs,
) -> tuple[Path, tuple[Path, Path], tuple[Path, Path], tuple[Path, Path], Path, Path]:
    """Shared stages from data_03 manifest through data_09 assemble."""
    g2p_csv_path = g2p_manifest(manifest_csv_path, dirs.data_04)
    hubert_paths = hubert_from_zip(zip_path, dirs.data_05)
    sv_paths = sv_from_zip(zip_path, dirs.data_06)
    semantic_paths = semantic_from_hubert_out(hubert_paths[1], dirs.data_07)
    wav32k_zip_path = wav32k_from_zip(zip_path, dirs.data_08)
    assemble_csv_path = assemble_manifest(
        zip_path,
        g2p_csv_path,
        dirs.data_09,
        hubert_data_dir=dirs.data_05,
        sv_data_dir=dirs.data_06,
        semantic_data_dir=dirs.data_07,
        wav32k_data_dir=dirs.data_08,
    )
    return g2p_csv_path, hubert_paths, sv_paths, semantic_paths, wav32k_zip_path, assemble_csv_path
