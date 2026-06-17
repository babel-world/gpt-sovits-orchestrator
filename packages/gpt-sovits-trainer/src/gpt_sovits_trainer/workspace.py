from __future__ import annotations

import re

PIPELINE_DRY_VOCAL = "dry-vocal"
PIPELINE_AI_HOBBYIST = "ai-hobbyist"
PIPELINE_CHOICES = (PIPELINE_DRY_VOCAL, PIPELINE_AI_HOBBYIST)

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


def validate_pipeline(pipeline: str) -> str:
    normalized = pipeline.strip()
    if normalized not in PIPELINE_CHOICES:
        raise ValueError(
            f"Invalid pipeline {pipeline!r}; expected one of: {', '.join(PIPELINE_CHOICES)}"
        )
    return normalized


def resolve_trainer_target(pipeline: str, speaker: str) -> tuple[str, str, str]:
    """Return (workspace, base_name, zip_stem) for `{pipeline}/{speaker}`."""
    pipeline_id = validate_pipeline(pipeline)
    slug = validate_speaker_slug(speaker)
    return f"{pipeline_id}/{slug}", slug, f"{slug}_slices"
