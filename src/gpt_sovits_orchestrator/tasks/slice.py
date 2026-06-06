from __future__ import annotations

import zipfile
from pathlib import Path

import httpx
from prefect import task

from gpt_sovits_orchestrator.config import ASR_SERVER_BASE_URL, SLICE_API_PATH


def _filename_from_content_disposition(header: str | None, fallback: str) -> str:
    if not header:
        return fallback

    for part in header.split(";"):
        part = part.strip()
        if part.startswith("filename="):
            value = part.removeprefix("filename=").strip().strip('"')
            if value:
                return value
    return fallback


@task(name="slice-audio", log_prints=True)
def slice_audio(
    audio_path: Path,
    output_dir: Path,
    *,
    base_url: str = ASR_SERVER_BASE_URL,
    timeout: float = 600.0,
) -> Path:
    """Upload raw audio to ``POST /api/audio/slice`` and save the returned ZIP."""
    audio_path = audio_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    url = f"{base_url.rstrip('/')}{SLICE_API_PATH}"
    fallback_name = f"{audio_path.stem}_slices.zip"

    with audio_path.open("rb") as audio_file:
        files = {"file": (audio_path.name, audio_file, "application/octet-stream")}
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, files=files)

    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "zip" not in content_type and response.content[:2] != b"PK":
        raise ValueError(
            f"Expected ZIP response from {url}, got content-type={content_type!r}"
        )

    download_name = _filename_from_content_disposition(
        response.headers.get("content-disposition"),
        fallback_name,
    )
    zip_path = output_dir / download_name
    zip_path.write_bytes(response.content)

    with zipfile.ZipFile(zip_path, "r") as archive:
        if not archive.namelist():
            raise ValueError(f"Downloaded ZIP is empty: {zip_path}")

    print(f"Saved slice ZIP: {zip_path} ({len(response.content)} bytes)")
    return zip_path
