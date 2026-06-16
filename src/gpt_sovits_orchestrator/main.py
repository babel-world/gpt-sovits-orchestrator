from pathlib import Path

from prefect import flow

from gpt_sovits_orchestrator.config import (
    DATA_01_DIR,
    DATA_02_DIR,
)
from gpt_sovits_orchestrator.flow_common import orchestrator_tail
from gpt_sovits_orchestrator.tasks.slice import slice_audio
from gpt_sovits_orchestrator.tasks.transcribe import transcribe_slices


@flow(name="gpt-sovits-orchestrator")
def orchestrator_flow(
    audio_path: Path | None = None,
) -> tuple[Path, tuple[Path, Path], tuple[Path, Path], tuple[Path, Path], Path, Path]:
    source = audio_path or (DATA_01_DIR / "manbo.mp3")
    zip_path = slice_audio(source, DATA_02_DIR)
    csv_path = transcribe_slices(zip_path, DATA_03_DIR)
    return orchestrator_tail(zip_path, csv_path)


def main() -> None:
    g2p_csv_path, (hubert_in_npz, hubert_out_npz), (sv_in_npz, sv_out_npz), (
        semantic_in_npz,
        semantic_out_npz,
    ), wav32k_zip, assemble_csv = orchestrator_flow()
    print(f"data_04 ready: {g2p_csv_path}")
    print(f"data_05 ready: {hubert_in_npz}")
    print(f"data_05 ready: {hubert_out_npz}")
    print(f"data_06 ready: {sv_in_npz}")
    print(f"data_06 ready: {sv_out_npz}")
    print(f"data_07 ready: {semantic_in_npz}")
    print(f"data_07 ready: {semantic_out_npz}")
    print(f"data_08 ready: {wav32k_zip}")
    print(f"data_09 ready: {assemble_csv}")


if __name__ == "__main__":
    main()
