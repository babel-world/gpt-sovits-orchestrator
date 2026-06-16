from __future__ import annotations

import re

SPEAKER_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_speaker_slug(speaker: str) -> str:
    slug = speaker.strip()
    if not slug:
        raise ValueError("speaker slug must not be empty")
    if not SPEAKER_SLUG_RE.match(slug):
        raise ValueError(
            f"Invalid speaker slug {speaker!r}; expected lowercase ASCII like 'jinxi'"
        )
    return slug


def slices_zip_stem(speaker: str) -> str:
    return f"{validate_speaker_slug(speaker)}_slices"
