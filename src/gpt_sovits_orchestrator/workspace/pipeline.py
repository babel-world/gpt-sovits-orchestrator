from __future__ import annotations

PIPELINE_DRY_VOCAL = "dry-vocal"
PIPELINE_AI_HOBBYIST = "ai-hobbyist"

PIPELINE_CHOICES = (PIPELINE_DRY_VOCAL, PIPELINE_AI_HOBBYIST)


def pipeline_dir_name(pipeline: str) -> str:
    """Map CLI pipeline id to `.local/` subdirectory (kebab-case)."""
    normalized = pipeline.strip()
    if normalized not in PIPELINE_CHOICES:
        raise ValueError(
            f"Invalid pipeline {pipeline!r}; expected one of: {', '.join(PIPELINE_CHOICES)}"
        )
    return normalized
