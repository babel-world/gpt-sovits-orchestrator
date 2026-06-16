from __future__ import annotations

from pathlib import Path

from prefect import flow

from gpt_sovits_orchestrator.ai_hobbyist.workspace import AiHobbyistWorkspace
from gpt_sovits_orchestrator.config import AI_HOBBYIST_SPEAKER
from gpt_sovits_orchestrator.flow_common import orchestrator_tail
from gpt_sovits_orchestrator.tasks.ai_hobbyist_import import ai_hobbyist_import


def resolve_ai_hobbyist_speaker(speaker: str | None) -> str:
    resolved = (speaker or AI_HOBBYIST_SPEAKER or "").strip()
    if not resolved:
        raise ValueError(
            "AI-Hobbyist speaker slug is required. "
            "Pass speaker=... to the flow, set AI_HOBBYIST_SPEAKER in .env, "
            "or pass it as the first CLI argument."
        )
    return resolved


@flow(name="ai-hobbyist-orchestrator")
def ai_hobbyist_orchestrator_flow(
    speaker: str | None = None,
    *,
    inbox_dir: Path | None = None,
) -> tuple[Path, tuple[Path, Path], tuple[Path, Path], tuple[Path, Path], Path, Path]:
    slug = resolve_ai_hobbyist_speaker(speaker)
    workspace = AiHobbyistWorkspace.for_speaker(slug)
    workspace.ensure_exists()
    stage_dirs = workspace.pipeline_dirs
    resolved_inbox = inbox_dir or workspace.inbox
    zip_path, manifest_csv_path = ai_hobbyist_import(
        slug,
        inbox_dir=resolved_inbox,
        output_zip_dir=stage_dirs.data_02,
        output_manifest_dir=stage_dirs.data_03,
    )
    return orchestrator_tail(zip_path, manifest_csv_path, dirs=stage_dirs)
