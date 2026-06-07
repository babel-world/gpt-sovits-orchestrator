"""Shared path constants for trainer and repo `.local` data."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

TRAINER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TRAINER_ROOT.parents[1]
load_dotenv(TRAINER_ROOT / ".env", override=False)
LOCAL_DIR = REPO_ROOT / ".local"
DATA_09_DIR = LOCAL_DIR / "data_09"
DATA_10_DIR = LOCAL_DIR / "data_10"

VENDOR_ROOT = TRAINER_ROOT / "vendor"
GPT_SOVITS_ROOT = Path(os.environ.get("GPT_SOVITS_ROOT", VENDOR_ROOT / "GPT_SoVITS"))
MODELS_DIR = TRAINER_ROOT / ".models" / "pretrained"

DEFAULT_BASE_NAME = "manbo"
DEFAULT_ZIP_STEM = "manbo_slices"
EXP_NAME = DEFAULT_BASE_NAME

PRETRAINED_S1 = MODELS_DIR / "s1v3.ckpt"
PRETRAINED_S2G = MODELS_DIR / "v2Pro" / "s2Gv2Pro.pth"
PRETRAINED_S2D = MODELS_DIR / "v2Pro" / "s2Dv2Pro.pth"

HF_REPO_ID = "lj1995/GPT-SoVITS"


def assemble_manifest_path(base_name: str = DEFAULT_BASE_NAME) -> Path:
    return DATA_09_DIR / f"{base_name}_manifest_assemble.csv"


def modality_paths(zip_stem: str = DEFAULT_ZIP_STEM) -> dict[str, Path]:
    return {
        "hubert_out": LOCAL_DIR / "data_05" / f"{zip_stem}_hubert_out.npz",
        "sv_out": LOCAL_DIR / "data_06" / f"{zip_stem}_sv_out.npz",
        "semantic_out": LOCAL_DIR / "data_07" / f"{zip_stem}_semantic_out.npz",
        "wav32k": LOCAL_DIR / "data_08" / f"{zip_stem}_wav32k.zip",
    }


def ensure_data_dirs() -> None:
    DATA_10_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "v2Pro").mkdir(parents=True, exist_ok=True)
