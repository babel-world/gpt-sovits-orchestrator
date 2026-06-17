from __future__ import annotations

from pathlib import Path

from prefect import flow

from gpt_sovits_orchestrator.dry_vocal.inbox import resolve_dry_vocal_inbox
from gpt_sovits_orchestrator.flow_common import orchestrator_tail
from gpt_sovits_orchestrator.tasks.slice import slice_audio
from gpt_sovits_orchestrator.tasks.transcribe import transcribe_slices
from gpt_sovits_orchestrator.workspace import PIPELINE_DRY_VOCAL, SpeakerWorkspace


@flow(name="dry-vocal-orchestrator")
def dry_vocal_orchestrator_flow(
    speaker: str,
) -> tuple[Path, tuple[Path, Path], tuple[Path, Path], tuple[Path, Path], Path, Path]:
    workspace = SpeakerWorkspace.create(PIPELINE_DRY_VOCAL, speaker)
    workspace.ensure_exists()
    stage_dirs = workspace.pipeline_dirs
    source = resolve_dry_vocal_inbox(workspace.inbox)
    zip_path = slice_audio(source, stage_dirs.data_02)
    csv_path = transcribe_slices(zip_path, stage_dirs.data_03)
    return orchestrator_tail(zip_path, csv_path, dirs=stage_dirs)
