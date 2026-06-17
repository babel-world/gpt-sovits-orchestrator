"""GPT-SoVITS s1/s2 training entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gpt_sovits_trainer.download_models import ensure_pretrained_models
from gpt_sovits_trainer.env import (
    TRAINER_BATCH_SIZE,
    TRAINER_BASE_NAME,
    TRAINER_S1_EPOCHS,
    TRAINER_S2_EPOCHS,
    TRAINER_WORKSPACE,
    TRAINER_ZIP_STEM,
)
from gpt_sovits_trainer.paths import (
    assemble_manifest_path,
    ensure_data_dirs,
    modality_paths,
    resolve_trainer_layout,
)
from gpt_sovits_trainer.stores import ModalityStores
from gpt_sovits_trainer.train_s1 import train_s1
from gpt_sovits_trainer.train_s2 import train_s2
from gpt_sovits_trainer.workspace import PIPELINE_CHOICES, resolve_trainer_target


def trainer_flow(
    *,
    base_name: str = TRAINER_BASE_NAME,
    zip_stem: str = TRAINER_ZIP_STEM,
    workspace: str = TRAINER_WORKSPACE,
    s1_epochs: int = TRAINER_S1_EPOCHS,
    s2_epochs: int = TRAINER_S2_EPOCHS,
    batch_size: int = TRAINER_BATCH_SIZE,
) -> tuple[Path, Path]:
    layout = ensure_data_dirs(layout=resolve_trainer_layout(workspace))
    ensure_pretrained_models()

    assemble_csv = assemble_manifest_path(base_name, layout=layout)
    if not assemble_csv.is_file():
        raise FileNotFoundError(
            f"Assemble manifest not found: {assemble_csv}. Run orchestrator through data_09 first."
        )

    paths = modality_paths(zip_stem, layout=layout)
    stores = ModalityStores(
        hubert_out_npz=paths["hubert_out"],
        sv_out_npz=paths["sv_out"],
        semantic_out_npz=paths["semantic_out"],
        wav32k_zip=paths["wav32k"],
    )
    try:
        ckpt = train_s1(
            assemble_csv,
            stores,
            epochs=s1_epochs,
            batch_size=batch_size,
            exp_name=base_name,
            weights_dir=layout.weights_dir,
        )
        pth = train_s2(
            assemble_csv,
            stores,
            epochs=s2_epochs,
            batch_size=batch_size,
            exp_name=base_name,
            weights_dir=layout.weights_dir,
        )
    finally:
        stores.close()
    return ckpt, pth


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpt-sovits-trainer",
        description="GPT-SoVITS s1/s2 training (usually invoked by gpt-sovits-orchestrator)",
    )
    parser.add_argument(
        "pipeline",
        nargs="?",
        choices=PIPELINE_CHOICES,
        help="pipeline id (dry-vocal or ai-hobbyist)",
    )
    parser.add_argument(
        "speaker",
        nargs="?",
        help="speaker slug (e.g. manbo, jinxi)",
    )
    return parser


def _resolve_training_target(
    pipeline: str | None,
    speaker: str | None,
) -> tuple[str, str, str]:
    if pipeline and speaker:
        return resolve_trainer_target(pipeline, speaker)
    if pipeline or speaker:
        raise SystemExit("pipeline and speaker must be provided together")
    if TRAINER_WORKSPACE.strip():
        return TRAINER_WORKSPACE.strip(), TRAINER_BASE_NAME, TRAINER_ZIP_STEM
    raise SystemExit(
        "usage: gpt-sovits-trainer {dry-vocal|ai-hobbyist} speaker\n"
        "or set TRAINER_WORKSPACE in repo root .env (legacy)"
    )


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    workspace, base_name, zip_stem = _resolve_training_target(args.pipeline, args.speaker)
    ckpt, pth = trainer_flow(
        workspace=workspace,
        base_name=base_name,
        zip_stem=zip_stem,
    )
    print(f"data_10 ready: {ckpt}")
    print(f"data_10 ready: {pth}")


if __name__ == "__main__":
    main(sys.argv[1:])
