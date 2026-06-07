"""Bootstrap vendor GPT-SoVITS imports."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from gpt_sovits_trainer.paths import GPT_SOVITS_ROOT, VENDOR_ROOT


def bootstrap_vendor() -> Path:
    gpt_root = GPT_SOVITS_ROOT.resolve()
    vendor_root = VENDOR_ROOT.resolve()
    if not gpt_root.is_dir():
        raise FileNotFoundError(
            f"GPT-SoVITS vendor code not found at {gpt_root}. "
            "Run scripts/sync_vendor.ps1 first."
        )

    for path in (vendor_root, gpt_root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

    os.environ.setdefault("version", "v2Pro")
    os.environ.setdefault("hz", "25hz")
    os.chdir(gpt_root)
    return gpt_root
