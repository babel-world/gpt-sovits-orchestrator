from __future__ import annotations

import re
import zipfile
from pathlib import Path

import httpx
import numpy as np
from prefect import task

from gpt_sovits_orchestrator.config import (
    ASR_SERVER_BASE_URL,
    HUBERT_API_TIMEOUT_S,
    HUBERT_EXTRACT_PATH,
    HUBERT_START_PATH,
    HUBERT_STOP_PATH,
)
from gpt_sovits_orchestrator.hubert.preprocess import wav_bytes_to_hubert_input


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


def _parse_content_disposition(header: str | None) -> str | None:
    if not header:
        return None
    match = re.search(r'filename="?([^";\n]+)"?', header)
    return match.group(1) if match else None


def _check_server(base_url: str) -> None:
    try:
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            response = client.get("/")
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Cannot reach asr-server ({base_url}), ensure it is running: {exc}"
        ) from exc


def _api_start(client: httpx.Client, start_path: str) -> None:
    response = client.post(start_path)
    response.raise_for_status()
    body = response.json()
    print(f"[start] loaded={body.get('loaded')} {body.get('message', '')}")


def _api_stop(client: httpx.Client, stop_path: str) -> None:
    response = client.post(stop_path)
    response.raise_for_status()
    body = response.json()
    print(f"[stop] loaded={body.get('loaded')} {body.get('message', '')}")


def _extract_one(
    input_path: Path,
    output_dir: Path,
    *,
    client: httpx.Client,
    extract_path: str,
    skip_existing: bool,
) -> Path | None:
    stem = input_path.stem
    default_name = f"{stem}_features.npy"
    out_path = output_dir / default_name

    if skip_existing and out_path.is_file():
        print(f"SKIP {input_path.name} (output exists)")
        return None

    npy_bytes = input_path.read_bytes()
    response = client.post(
        extract_path,
        files={"file": (f"{stem}.npy", npy_bytes, "application/octet-stream")},
    )
    if response.status_code >= 400:
        detail = response.text[:200]
        raise RuntimeError(f"HTTP {response.status_code}: {detail}")

    download_name = _parse_content_disposition(
        response.headers.get("content-disposition")
    ) or default_name
    out_path = output_dir / download_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(response.content)
    return out_path


@task(name="prepare-hubert-inputs", log_prints=True)
def prepare_hubert_inputs(
    zip_path: Path,
    output_dir: Path,
) -> list[Path]:
    """Convert slice WAV entries in a ZIP to HuBERT input .npy files."""
    zip_path = zip_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.is_file():
        raise FileNotFoundError(f"Slice ZIP not found: {zip_path}")

    wav_names = _list_wav_entries(zip_path)
    outputs: list[Path] = []
    skipped = 0
    total = len(wav_names)

    with zipfile.ZipFile(zip_path, "r") as archive:
        for index, name in enumerate(wav_names, start=1):
            stem = Path(name).name
            stem_id = Path(stem).stem
            out_path = output_dir / f"{stem_id}.npy"
            arr = wav_bytes_to_hubert_input(archive.read(name), stem)
            if arr is None:
                skipped += 1
                print(f"SKIP [{index}/{total}] {stem}")
                continue
            np.save(out_path, arr)
            outputs.append(out_path)
            print(f"OK   [{index}/{total}] {stem} -> {out_path.name}")

    print(f"Prepared HuBERT inputs: {len(outputs)} ok, {skipped} skipped")
    if not outputs:
        raise RuntimeError(f"No HuBERT inputs generated from ZIP: {zip_path}")
    return outputs


@task(name="extract-hubert-features", log_prints=True)
def extract_hubert_features(
    inputs_dir: Path,
    output_dir: Path,
    *,
    base_url: str = ASR_SERVER_BASE_URL,
    skip_existing: bool = True,
    preload: bool = True,
) -> list[Path]:
    """Extract HuBERT features via asr-server API for all input .npy files."""
    inputs_dir = inputs_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not inputs_dir.is_dir():
        raise FileNotFoundError(f"HuBERT inputs directory not found: {inputs_dir}")

    npy_files = sorted(inputs_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in: {inputs_dir}")

    _check_server(base_url)
    successes: list[Path] = []
    failures: list[str] = []

    with httpx.Client(base_url=base_url, timeout=HUBERT_API_TIMEOUT_S) as client:
        if preload:
            _api_start(client, HUBERT_START_PATH)
        try:
            for npy_path in npy_files:
                try:
                    out_path = _extract_one(
                        npy_path,
                        output_dir,
                        client=client,
                        extract_path=HUBERT_EXTRACT_PATH,
                        skip_existing=skip_existing,
                    )
                    if out_path is not None:
                        successes.append(out_path)
                        print(f"OK   {npy_path.name} -> {out_path.name}")
                except RuntimeError as exc:
                    failures.append(npy_path.name)
                    print(f"FAIL {npy_path.name}: {exc}")
        finally:
            if preload:
                _api_stop(client, HUBERT_STOP_PATH)

    print(f"Extracted HuBERT features: {len(successes)} ok, {len(failures)} failed")
    if failures:
        raise RuntimeError(f"HuBERT extraction failed for: {', '.join(failures)}")
    return successes


@task(name="hubert-from-zip", log_prints=True)
def hubert_from_zip(
    zip_path: Path,
    data_dir: Path,
    *,
    base_url: str = ASR_SERVER_BASE_URL,
    skip_existing: bool = True,
    preload: bool = True,
) -> tuple[list[Path], list[Path]]:
    """Prepare HuBERT inputs from slice ZIP and extract features to data_05."""
    inputs_dir = data_dir / "hubert_inputs"
    outputs_dir = data_dir / "hubert_outputs"
    input_paths = prepare_hubert_inputs(zip_path, inputs_dir)
    output_paths = extract_hubert_features(
        inputs_dir,
        outputs_dir,
        base_url=base_url,
        skip_existing=skip_existing,
        preload=preload,
    )
    return input_paths, output_paths
