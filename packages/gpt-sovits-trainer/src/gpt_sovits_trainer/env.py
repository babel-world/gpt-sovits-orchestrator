"""Environment variables for trainer (loaded from repo root `.env`)."""

from __future__ import annotations

import os

from gpt_sovits_trainer import paths as _paths  # noqa: F401 — loads repo `.env`


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw is not None else default


TRAINER_S1_EPOCHS = _env_int("TRAINER_S1_EPOCHS", 2)
TRAINER_S2_EPOCHS = _env_int("TRAINER_S2_EPOCHS", 2)
TRAINER_BATCH_SIZE = _env_int("TRAINER_BATCH_SIZE", 4)
