from pathlib import Path

from prefect import flow

from gpt_sovits_orchestrator.config import (
    DATA_01_DIR,
    DATA_02_DIR,
    DATA_03_DIR,
    DATA_04_DIR,
    DATA_05_DIR,
)
from gpt_sovits_orchestrator.tasks.g2p import g2p_manifest
from gpt_sovits_orchestrator.tasks.hubert import hubert_from_zip
from gpt_sovits_orchestrator.tasks.slice import slice_audio
from gpt_sovits_orchestrator.tasks.transcribe import transcribe_slices


@flow(name="gpt-sovits-orchestrator")
def orchestrator_flow(audio_path: Path | None = None) -> tuple[Path, tuple[list[Path], list[Path]]]:
    source = audio_path or (DATA_01_DIR / "manbo.mp3")
    zip_path = slice_audio(source, DATA_02_DIR)
    csv_path = transcribe_slices(zip_path, DATA_03_DIR)
    g2p_csv_path = g2p_manifest(csv_path, DATA_04_DIR)
    hubert_result = hubert_from_zip(zip_path, DATA_05_DIR)
    return g2p_csv_path, hubert_result


def main() -> None:
    g2p_csv_path, (_, hubert_outputs) = orchestrator_flow()
    print(f"data_04 ready: {g2p_csv_path}")
    print(f"data_05 ready: {len(hubert_outputs)} feature files in {DATA_05_DIR / 'hubert_outputs'}")


if __name__ == "__main__":
    main()
