from __future__ import annotations

import shutil
import subprocess
import sys

from gpt_sovits_orchestrator.config import PROJECT_ROOT
from gpt_sovits_orchestrator.workspace import pipeline_dir_name, validate_speaker_slug

TRAINER_PROJECT_DIR = PROJECT_ROOT / "packages" / "gpt-sovits-trainer"


def run_trainer_subprocess(pipeline: str, speaker: str) -> None:
    """Run gpt-sovits-trainer in its own uv env (Python 3.10 + PyTorch)."""
    pipeline_id = pipeline_dir_name(pipeline)
    slug = validate_speaker_slug(speaker)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "Cannot find `uv` on PATH. Install uv or run trainer manually:\n"
            f"  cd {TRAINER_PROJECT_DIR}\n"
            f"  uv run gpt-sovits-trainer {pipeline_id} {slug}"
        )
    if not TRAINER_PROJECT_DIR.is_dir():
        raise FileNotFoundError(f"Trainer project not found: {TRAINER_PROJECT_DIR}")

    print(f"Starting trainer: {pipeline_id} {slug} …")
    subprocess.run(
        [uv, "run", "gpt-sovits-trainer", pipeline_id, slug],
        cwd=TRAINER_PROJECT_DIR,
        check=True,
    )
