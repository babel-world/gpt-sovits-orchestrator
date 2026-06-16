from __future__ import annotations

import io
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
    SERVER_HEALTH_CHECK_TIMEOUT_S,
)
from gpt_sovits_orchestrator.hubert.npz import hubert_npz_paths
from gpt_sovits_orchestrator.hubert.preprocess import wav_bytes_to_hubert_input
from gpt_sovits_orchestrator.utils.npz_stream import NpzStreamWriter


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


def _array_to_npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _check_server(base_url: str) -> None:
    try:
        with httpx.Client(base_url=base_url, timeout=SERVER_HEALTH_CHECK_TIMEOUT_S) as client:
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


def _extract_feature(
    stem: str,
    npy_bytes: bytes,
    *,
    client: httpx.Client,
    extract_path: str,
) -> np.ndarray:
    response = client.post(
        extract_path,
        files={"file": (f"{stem}.npy", npy_bytes, "application/octet-stream")},
    )
    if response.status_code >= 400:
        detail = response.text[:200]
        raise RuntimeError(f"HTTP {response.status_code}: {detail}")
    return np.load(io.BytesIO(response.content))


@task(name="prepare-hubert-inputs", log_prints=True)
def prepare_hubert_inputs(
    zip_path: Path,
    in_npz_path: Path,
) -> Path:
    """Convert slice WAV entries in a ZIP to a HuBERT input NPZ archive."""
    zip_path = zip_path.resolve()
    in_npz_path = in_npz_path.resolve()
    in_npz_path.parent.mkdir(parents=True, exist_ok=True)

    if not zip_path.is_file():
        raise FileNotFoundError(f"Slice ZIP not found: {zip_path}")

    wav_names = _list_wav_entries(zip_path)
    skipped = 0
    total = len(wav_names)
    written = 0

    with NpzStreamWriter(in_npz_path) as writer, zipfile.ZipFile(zip_path, "r") as archive:
        for index, name in enumerate(wav_names, start=1):
            wav_key = Path(name).name
            arr = wav_bytes_to_hubert_input(archive.read(name), wav_key)
            if arr is None:
                skipped += 1
                print(f"SKIP [{index}/{total}] {wav_key}")
                continue
            writer.add(wav_key, arr)
            written += 1
            print(f"OK   [{index}/{total}] {wav_key}")

    if written == 0:
        raise RuntimeError(f"No HuBERT inputs generated from ZIP: {zip_path}")

    print(f"Saved HuBERT input NPZ: {in_npz_path} ({written} arrays, {skipped} skipped)")
    return in_npz_path


@task(name="extract-hubert-features", log_prints=True)
def extract_hubert_features(
    in_npz_path: Path,
    out_npz_path: Path,
    *,
    base_url: str = ASR_SERVER_BASE_URL,
    preload: bool = True,
) -> Path:
    """Extract HuBERT features via asr-server API and save as NPZ."""
    in_npz_path = in_npz_path.resolve()
    out_npz_path = out_npz_path.resolve()
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_npz_path.is_file():
        raise FileNotFoundError(f"HuBERT input NPZ not found: {in_npz_path}")

    _check_server(base_url)
    failures: list[str] = []
    written = 0

    with np.load(in_npz_path) as inputs, NpzStreamWriter(out_npz_path) as writer, httpx.Client(
        base_url=base_url, timeout=HUBERT_API_TIMEOUT_S
    ) as client:
        if preload:
            _api_start(client, HUBERT_START_PATH)
        try:
            for wav_key in sorted(inputs.files):
                stem = Path(wav_key).stem
                try:
                    feature = _extract_feature(
                        stem,
                        _array_to_npy_bytes(inputs[wav_key]),
                        client=client,
                        extract_path=HUBERT_EXTRACT_PATH,
                    )
                    writer.add(wav_key, feature)
                    written += 1
                    print(f"OK   {wav_key} -> feature {feature.shape}")
                except RuntimeError as exc:
                    failures.append(wav_key)
                    print(f"FAIL {wav_key}: {exc}")
        finally:
            if preload:
                _api_stop(client, HUBERT_STOP_PATH)

    print(f"Extracted HuBERT features: {written} ok, {len(failures)} failed")
    if failures:
        raise RuntimeError(f"HuBERT extraction failed for: {', '.join(failures)}")

    print(f"Saved HuBERT output NPZ: {out_npz_path} ({written} arrays)")
    return out_npz_path


@task(name="hubert-from-zip", log_prints=True)
def hubert_from_zip(
    zip_path: Path,
    data_dir: Path,
    *,
    base_url: str = ASR_SERVER_BASE_URL,
    preload: bool = True,
) -> tuple[Path, Path]:
    """Prepare HuBERT input NPZ from slice ZIP and extract features to output NPZ."""
    zip_path = zip_path.resolve()
    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    in_npz_path, out_npz_path = hubert_npz_paths(zip_path, data_dir)
    prepare_hubert_inputs(zip_path, in_npz_path)
    extract_hubert_features(
        in_npz_path,
        out_npz_path,
        base_url=base_url,
        preload=preload,
    )
    return in_npz_path, out_npz_path
