from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import httpx
from prefect import task

from gpt_sovits_orchestrator.config import (
    ASR_SERVER_BASE_URL,
    MANIFEST_FIELDS,
    TRANSCRIBE_API_PATH,
)
from gpt_sovits_orchestrator.utils.slice_filename import parse_slice_chunk_filename


def _list_wav_entries(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".wav") and not name.endswith("/")
        ]
    if not names:
        raise ValueError(f"No WAV entries found in ZIP: {zip_path}")
    return sorted(names, key=lambda name: Path(name).name)


@task(name="transcribe-slices", log_prints=True)
def transcribe_slices(
    zip_path: Path,
    output_dir: Path,
    *,
    base_url: str = ASR_SERVER_BASE_URL,
    timeout: float = 600.0,
) -> Path:
    """Transcribe each WAV in a slice ZIP via ``POST /api/transcribe`` and write manifest CSV."""
    zip_path = zip_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.is_file():
        raise FileNotFoundError(f"Slice ZIP not found: {zip_path}")

    wav_names = _list_wav_entries(zip_path)
    base_names = {parse_slice_chunk_filename(Path(name).name).base_name for name in wav_names}
    if len(base_names) != 1:
        raise ValueError(
            f"Expected a single base_name across all WAV entries, got: {sorted(base_names)}"
        )
    base_name = next(iter(base_names))
    csv_path = output_dir / f"{base_name}_manifest.csv"

    url = f"{base_url.rstrip('/')}{TRANSCRIBE_API_PATH}"
    rows: list[dict[str, str | float]] = []
    total = len(wav_names)

    with zipfile.ZipFile(zip_path, "r") as archive, httpx.Client(timeout=timeout) as client:
        for index, name in enumerate(wav_names, start=1):
            wav_bytes = archive.read(name)
            files = {"file": (name, io.BytesIO(wav_bytes), "audio/wav")}
            response = client.post(url, files=files)
            response.raise_for_status()
            body = response.json()

            rows.append(
                {
                    "filename": Path(name).name,
                    "speaker": base_name,
                    "language": body["language"],
                    "text": body["transcribedText"],
                    "probability": body["languageProbability"],
                }
            )
            print(f"[{index}/{total}] transcribed {Path(name).name}")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved manifest CSV: {csv_path} ({len(rows)} rows)")
    return csv_path
