"""Shared path constants for trainer and repo `.local` data."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

TRAINER_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TRAINER_ROOT.parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)
LOCAL_DIR = REPO_ROOT / ".local"
DATA_09_DIR = LOCAL_DIR / "data_09"
DATA_10_DIR = LOCAL_DIR / "data_10"

VENDOR_ROOT = TRAINER_ROOT / "vendor"
GPT_SOVITS_ROOT = Path(os.environ.get("GPT_SOVITS_ROOT", VENDOR_ROOT / "GPT_SoVITS"))
MODELS_DIR = TRAINER_ROOT / ".models" / "pretrained"

DEFAULT_BASE_NAME = "manbo"
DEFAULT_ZIP_STEM = "manbo_slices"

PRETRAINED_S1 = MODELS_DIR / "s1v3.ckpt"
PRETRAINED_S2G = MODELS_DIR / "v2Pro" / "s2Gv2Pro.pth"
PRETRAINED_S2D = MODELS_DIR / "v2Pro" / "s2Dv2Pro.pth"

HF_REPO_ID = "lj1995/GPT-SoVITS"


@dataclass(frozen=True)
class TrainerLayout:
    assemble_dir: Path
    weights_dir: Path
    hubert_dir: Path
    sv_dir: Path
    semantic_dir: Path
    wav32k_dir: Path


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def resolve_trainer_layout(workspace: str = "") -> TrainerLayout:
    """Resolve modality dirs; legacy manbo uses `.local/data_05`…`data_10`."""
    relative = workspace.strip()
    if relative:
        data = (LOCAL_DIR / relative / "data").resolve()
        return TrainerLayout(
            assemble_dir=data / "09",
            weights_dir=data / "10",
            hubert_dir=data / "05",
            sv_dir=data / "06",
            semantic_dir=data / "07",
            wav32k_dir=data / "08",
        )
    return TrainerLayout(
        assemble_dir=DATA_09_DIR,
        weights_dir=DATA_10_DIR,
        hubert_dir=LOCAL_DIR / "data_05",
        sv_dir=LOCAL_DIR / "data_06",
        semantic_dir=LOCAL_DIR / "data_07",
        wav32k_dir=LOCAL_DIR / "data_08",
    )


TRAINER_WORKSPACE = _env_str("TRAINER_WORKSPACE", "")


def assemble_manifest_path(base_name: str = DEFAULT_BASE_NAME, *, layout: TrainerLayout | None = None) -> Path:
    resolved = layout or resolve_trainer_layout(TRAINER_WORKSPACE)
    return resolved.assemble_dir / f"{base_name}_manifest_assemble.csv"


def modality_paths(zip_stem: str = DEFAULT_ZIP_STEM, *, layout: TrainerLayout | None = None) -> dict[str, Path]:
    resolved = layout or resolve_trainer_layout(TRAINER_WORKSPACE)
    return {
        "hubert_out": resolved.hubert_dir / f"{zip_stem}_hubert_out.npz",
        "sv_out": resolved.sv_dir / f"{zip_stem}_sv_out.npz",
        "semantic_out": resolved.semantic_dir / f"{zip_stem}_semantic_out.npz",
        "wav32k": resolved.wav32k_dir / f"{zip_stem}_wav32k.zip",
    }


def ensure_data_dirs(*, layout: TrainerLayout | None = None) -> TrainerLayout:
    resolved = layout or resolve_trainer_layout(TRAINER_WORKSPACE)
    resolved.weights_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "v2Pro").mkdir(parents=True, exist_ok=True)
    return resolved
