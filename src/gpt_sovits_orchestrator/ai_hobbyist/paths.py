from __future__ import annotations

from pathlib import Path

from gpt_sovits_orchestrator.ai_hobbyist.workspace import AiHobbyistWorkspace


def ai_hobbyist_inbox_dir(speaker: str) -> Path:
    return AiHobbyistWorkspace.for_speaker(speaker).inbox


def ai_hobbyist_workspace(speaker: str) -> AiHobbyistWorkspace:
    return AiHobbyistWorkspace.for_speaker(speaker)
