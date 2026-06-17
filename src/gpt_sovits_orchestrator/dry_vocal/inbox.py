from __future__ import annotations

from pathlib import Path

AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"})


def resolve_dry_vocal_inbox(inbox_dir: Path) -> Path:
    """Return the single audio file in a dry-vocal inbox directory."""
    inbox_dir = inbox_dir.resolve()
    if not inbox_dir.is_dir():
        raise FileNotFoundError(f"dry-vocal inbox not found: {inbox_dir}")

    entries = sorted(
        (path for path in inbox_dir.iterdir() if path.is_file()),
        key=lambda path: path.name.lower(),
    )
    if not entries:
        raise ValueError(
            f"{inbox_dir} 需要恰好 1 个音频文件，当前找到 0 个。\n"
            f"请检查 {inbox_dir}"
        )

    audio_files = [path for path in entries if path.suffix.lower() in AUDIO_SUFFIXES]
    non_audio = [path.name for path in entries if path.suffix.lower() not in AUDIO_SUFFIXES]

    if non_audio:
        listed = ", ".join(path.name for path in entries)
        raise ValueError(
            f"{inbox_dir} 只能包含 1 个常见音频文件，"
            f"发现非音频文件: {', '.join(non_audio)}（全部文件: {listed}）。\n"
            f"请检查 {inbox_dir}"
        )

    if len(audio_files) != 1:
        listed = ", ".join(path.name for path in audio_files) or "(无)"
        raise ValueError(
            f"{inbox_dir} 需要恰好 1 个音频文件，当前找到 {len(audio_files)} 个: {listed}。\n"
            f"请检查 {inbox_dir}"
        )

    return audio_files[0]
