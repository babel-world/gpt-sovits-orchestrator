from gpt_sovits_orchestrator.workspace.pipeline import (
    PIPELINE_AI_HOBBYIST,
    PIPELINE_CHOICES,
    PIPELINE_DRY_VOCAL,
    pipeline_dir_name,
)
from gpt_sovits_orchestrator.workspace.slug import (
    SPEAKER_SLUG_RE,
    slices_zip_stem,
    validate_speaker_slug,
)
from gpt_sovits_orchestrator.workspace.speaker_workspace import SpeakerWorkspace

__all__ = [
    "PIPELINE_AI_HOBBYIST",
    "PIPELINE_CHOICES",
    "PIPELINE_DRY_VOCAL",
    "SPEAKER_SLUG_RE",
    "SpeakerWorkspace",
    "pipeline_dir_name",
    "slices_zip_stem",
    "validate_speaker_slug",
]
