from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gpt_sovits_orchestrator.config import LOCAL_DIR
from gpt_sovits_orchestrator.pipeline_dirs import PipelineStageDirs
from gpt_sovits_orchestrator.workspace.pipeline import pipeline_dir_name
from gpt_sovits_orchestrator.workspace.slug import validate_speaker_slug


@dataclass(frozen=True)
class SpeakerWorkspace:
    """Per-speaker workspace under `.local/{pipeline}/{speaker}/`."""

    pipeline: str
    speaker: str

    @classmethod
    def create(cls, pipeline: str, speaker: str) -> SpeakerWorkspace:
        pipeline_id = pipeline_dir_name(pipeline)
        slug = validate_speaker_slug(speaker)
        return cls(pipeline=pipeline_id, speaker=slug)

    @property
    def root(self) -> Path:
        return LOCAL_DIR / self.pipeline / self.speaker

    @property
    def inbox(self) -> Path:
        return self.root / "inbox"

    @property
    def pipeline_dirs(self) -> PipelineStageDirs:
        return PipelineStageDirs.under_workspace_data(self.root)

    def ensure_exists(self) -> None:
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.pipeline_dirs.ensure_exists()
