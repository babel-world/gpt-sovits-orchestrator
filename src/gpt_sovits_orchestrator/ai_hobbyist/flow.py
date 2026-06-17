from __future__ import annotations

from pathlib import Path

from prefect import flow

from gpt_sovits_orchestrator.flow_common import orchestrator_tail
from gpt_sovits_orchestrator.tasks.ai_hobbyist_import import ai_hobbyist_import
from gpt_sovits_orchestrator.workspace import PIPELINE_AI_HOBBYIST, SpeakerWorkspace, validate_speaker_slug


@flow(name="ai-hobbyist-orchestrator")
def ai_hobbyist_orchestrator_flow(
    speaker: str,
) -> tuple[Path, tuple[Path, Path], tuple[Path, Path], tuple[Path, Path], Path, Path]:
    slug = validate_speaker_slug(speaker)
    workspace = SpeakerWorkspace.create(PIPELINE_AI_HOBBYIST, slug)
    workspace.ensure_exists()
    stage_dirs = workspace.pipeline_dirs
    zip_path, manifest_csv_path = ai_hobbyist_import(
        slug,
        inbox_dir=workspace.inbox,
        output_zip_dir=stage_dirs.data_02,
        output_manifest_dir=stage_dirs.data_03,
    )
    return orchestrator_tail(zip_path, manifest_csv_path, dirs=stage_dirs)
