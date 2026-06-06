from pathlib import Path

from prefect import flow

from gpt_sovits_orchestrator.config import DATA_01_DIR, DATA_02_DIR
from gpt_sovits_orchestrator.tasks.slice import slice_audio


@flow(name="gpt-sovits-orchestrator")
def orchestrator_flow(audio_path: Path | None = None) -> Path:
    source = audio_path or (DATA_01_DIR / "manbo.mp3")
    return slice_audio(source, DATA_02_DIR)


def main() -> None:
    zip_path = orchestrator_flow()
    print(f"data_02 ready: {zip_path}")


if __name__ == "__main__":
    main()
