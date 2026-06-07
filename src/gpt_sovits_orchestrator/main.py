from pathlib import Path

from prefect import flow

from gpt_sovits_orchestrator.config import (
    DATA_01_DIR,
    DATA_02_DIR,
    DATA_03_DIR,
    DATA_04_DIR,
    DATA_05_DIR,
    DATA_06_DIR,
    DATA_07_DIR,
    DATA_08_DIR,
)
from gpt_sovits_orchestrator.tasks.g2p import g2p_manifest
from gpt_sovits_orchestrator.tasks.hubert import hubert_from_zip
from gpt_sovits_orchestrator.tasks.semantic import semantic_from_hubert_out
from gpt_sovits_orchestrator.tasks.slice import slice_audio
from gpt_sovits_orchestrator.tasks.sv import sv_from_zip
from gpt_sovits_orchestrator.tasks.transcribe import transcribe_slices
from gpt_sovits_orchestrator.tasks.wav32k import wav32k_from_zip


@flow(name="gpt-sovits-orchestrator")
def orchestrator_flow(
    audio_path: Path | None = None,
) -> tuple[Path, tuple[Path, Path], tuple[Path, Path], tuple[Path, Path], Path]:
    source = audio_path or (DATA_01_DIR / "manbo.mp3")
    zip_path = slice_audio(source, DATA_02_DIR)
    csv_path = transcribe_slices(zip_path, DATA_03_DIR)
    g2p_csv_path = g2p_manifest(csv_path, DATA_04_DIR)
    hubert_paths = hubert_from_zip(zip_path, DATA_05_DIR)
    sv_paths = sv_from_zip(zip_path, DATA_06_DIR)
    semantic_paths = semantic_from_hubert_out(hubert_paths[1], DATA_07_DIR)
    wav32k_zip_path = wav32k_from_zip(zip_path, DATA_08_DIR)
    return g2p_csv_path, hubert_paths, sv_paths, semantic_paths, wav32k_zip_path


def main() -> None:
    g2p_csv_path, (hubert_in_npz, hubert_out_npz), (sv_in_npz, sv_out_npz), (
        semantic_in_npz,
        semantic_out_npz,
    ), wav32k_zip = orchestrator_flow()
    print(f"data_04 ready: {g2p_csv_path}")
    print(f"data_05 ready: {hubert_in_npz}")
    print(f"data_05 ready: {hubert_out_npz}")
    print(f"data_06 ready: {sv_in_npz}")
    print(f"data_06 ready: {sv_out_npz}")
    print(f"data_07 ready: {semantic_in_npz}")
    print(f"data_07 ready: {semantic_out_npz}")
    print(f"data_08 ready: {wav32k_zip}")


if __name__ == "__main__":
    main()
