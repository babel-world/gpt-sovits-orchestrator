"""GPT-SoVITS s1/s2 training entrypoint."""

from __future__ import annotations

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


def main() -> None:
    ckpt, pth = trainer_flow()
    print(f"data_10 ready: {ckpt}")
    print(f"data_10 ready: {pth}")


if __name__ == "__main__":
    main()
