"""Download v2Pro pretrained weights into `.models/pretrained/`."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

from gpt_sovits_trainer.paths import (
    HF_REPO_ID,
    MODELS_DIR,
    PRETRAINED_S1,
    PRETRAINED_S2D,
    PRETRAINED_S2G,
    ensure_data_dirs,
)


def _download(filename: str, target: Path) -> Path:
    if target.is_file():
        print(f"exists: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN") or None
    print(f"downloading {HF_REPO_ID}/{filename} -> {target.parent}")
    cached = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        local_dir=str(MODELS_DIR),
        token=token,
    )
    cached_path = Path(cached)
    if cached_path.resolve() != target.resolve():
        target.write_bytes(cached_path.read_bytes())
    print(f"ready: {target}")
    return target


def ensure_pretrained_models() -> tuple[Path, Path, Path]:
    ensure_data_dirs()
    s1 = _download("s1v3.ckpt", PRETRAINED_S1)
    s2g = _download("v2Pro/s2Gv2Pro.pth", PRETRAINED_S2G)
    s2d = _download("v2Pro/s2Dv2Pro.pth", PRETRAINED_S2D)
    return s1, s2g, s2d


def main() -> None:
    ensure_pretrained_models()
