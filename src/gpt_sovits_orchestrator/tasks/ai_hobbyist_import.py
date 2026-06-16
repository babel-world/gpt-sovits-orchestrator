from __future__ import annotations

from pathlib import Path

from prefect import task

from gpt_sovits_orchestrator.ai_hobbyist.import_dataset import import_ai_hobbyist_dataset
from gpt_sovits_orchestrator.ai_hobbyist.workspace import AiHobbyistWorkspace
from gpt_sovits_orchestrator.config import (
    AI_HOBBYIST_DEFAULT_LANGUAGE,
    AI_HOBBYIST_DEFAULT_PROBABILITY,
)


@task(name="ai-hobbyist-import", log_prints=True)
def ai_hobbyist_import(
    speaker: str,
    *,
    inbox_dir: Path | None = None,
    output_zip_dir: Path | None = None,
    output_manifest_dir: Path | None = None,
    language: str = AI_HOBBYIST_DEFAULT_LANGUAGE,
    probability: float = AI_HOBBYIST_DEFAULT_PROBABILITY,
) -> tuple[Path, Path]:
    """Import wav+lab pairs from AI-Hobbyist inbox into workspace data/02 and data/03."""
    workspace = AiHobbyistWorkspace.for_speaker(speaker)
    stage_dirs = workspace.pipeline_dirs
    resolved_inbox = (inbox_dir or workspace.inbox).resolve()
    zip_path, manifest_path, _summary = import_ai_hobbyist_dataset(
        resolved_inbox,
        speaker,
        output_zip_dir=output_zip_dir or stage_dirs.data_02,
        output_manifest_dir=output_manifest_dir or stage_dirs.data_03,
        language=language,
        probability=probability,
    )
    return zip_path, manifest_path
