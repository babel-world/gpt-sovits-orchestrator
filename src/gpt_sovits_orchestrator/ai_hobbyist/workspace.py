from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gpt_sovits_orchestrator.ai_hobbyist.slug import validate_speaker_slug
from gpt_sovits_orchestrator.config import AI_HOBBYIST_INBOX_DIR, LOCAL_DIR
from gpt_sovits_orchestrator.pipeline_dirs import PipelineStageDirs


@dataclass(frozen=True)
class AiHobbyistWorkspace:
    """Isolated per-speaker workspace under `.local/ai_hobbyist/{speaker}/`."""

    speaker: str
    root: Path

    @classmethod
    def for_speaker(cls, speaker: str) -> AiHobbyistWorkspace:
        slug = validate_speaker_slug(speaker)
        return cls(speaker=slug, root=LOCAL_DIR / "ai_hobbyist" / slug)

    @property
    def inbox(self) -> Path:
        if AI_HOBBYIST_INBOX_DIR.strip():
            return Path(AI_HOBBYIST_INBOX_DIR).resolve()
        return self.root / "inbox"

    @property
    def pipeline_dirs(self) -> PipelineStageDirs:
        return PipelineStageDirs.under_workspace_data(self.root)

    def ensure_exists(self) -> None:
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.pipeline_dirs.ensure_exists()
