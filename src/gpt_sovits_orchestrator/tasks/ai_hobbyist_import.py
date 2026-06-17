from __future__ import annotations

from pathlib import Path

from prefect import task

from gpt_sovits_orchestrator.ai_hobbyist.import_dataset import import_ai_hobbyist_dataset
from gpt_sovits_orchestrator.config import (
    AI_HOBBYIST_DEFAULT_LANGUAGE,
    AI_HOBBYIST_DEFAULT_PROBABILITY,
)


@task(name="ai-hobbyist-import", log_prints=True)
def ai_hobbyist_import(
    speaker: str,
    *,
    inbox_dir: Path,
    output_zip_dir: Path,
    output_manifest_dir: Path,
    language: str = AI_HOBBYIST_DEFAULT_LANGUAGE,
    probability: float = AI_HOBBYIST_DEFAULT_PROBABILITY,
) -> tuple[Path, Path]:
    """Import wav+lab pairs from AI-Hobbyist inbox into workspace data/02 and data/03."""
    zip_path, manifest_path, _summary = import_ai_hobbyist_dataset(
        inbox_dir.resolve(),
        speaker,
        output_zip_dir=output_zip_dir,
        output_manifest_dir=output_manifest_dir,
        language=language,
        probability=probability,
    )
    return zip_path, manifest_path
