from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from gpt_sovits_orchestrator.dry_vocal.inbox import resolve_dry_vocal_inbox
from gpt_sovits_orchestrator.main import _build_parser
from gpt_sovits_orchestrator.workspace import (
    PIPELINE_DRY_VOCAL,
    SpeakerWorkspace,
    pipeline_dir_name,
)


def test_pipeline_dir_name_accepts_kebab_case() -> None:
    assert pipeline_dir_name("dry-vocal") == "dry-vocal"
    assert pipeline_dir_name("ai-hobbyist") == "ai-hobbyist"


def test_pipeline_dir_name_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Invalid pipeline"):
        pipeline_dir_name("legacy-manbo")


def test_speaker_workspace_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "gpt_sovits_orchestrator.workspace.speaker_workspace.LOCAL_DIR",
        tmp_path,
    )
    workspace = SpeakerWorkspace.create(PIPELINE_DRY_VOCAL, "manbo")
    assert workspace.root == tmp_path / "dry-vocal" / "manbo"
    assert workspace.inbox == tmp_path / "dry-vocal" / "manbo" / "inbox"
    assert workspace.pipeline_dirs.data_09 == tmp_path / "dry-vocal" / "manbo" / "data" / "09"


def test_resolve_dry_vocal_inbox_single_audio(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    audio = inbox / "manbo.mp3"
    audio.write_bytes(b"fake")
    assert resolve_dry_vocal_inbox(inbox) == audio


def test_resolve_dry_vocal_inbox_empty(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    with pytest.raises(ValueError, match="恰好 1 个音频"):
        resolve_dry_vocal_inbox(inbox)


def test_resolve_dry_vocal_inbox_multiple_audio(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "a.wav").write_bytes(b"a")
    (inbox / "b.wav").write_bytes(b"b")
    with pytest.raises(ValueError, match="恰好 1 个音频"):
        resolve_dry_vocal_inbox(inbox)


def test_resolve_dry_vocal_inbox_non_audio_file(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "manbo.mp3").write_bytes(b"a")
    (inbox / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="非音频文件"):
        resolve_dry_vocal_inbox(inbox)


def test_cli_parser_requires_two_args() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_cli_parser_rejects_invalid_pipeline() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["legacy", "manbo"])


def test_cli_parser_accepts_valid_args() -> None:
    parser = _build_parser()
    args = parser.parse_args(["dry-vocal", "manbo"])
    assert args.pipeline == "dry-vocal"
    assert args.speaker == "manbo"
