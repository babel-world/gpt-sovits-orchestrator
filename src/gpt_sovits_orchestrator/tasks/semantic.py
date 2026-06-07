from __future__ import annotations

import io
from pathlib import Path

import httpx
import numpy as np
from prefect import task

from gpt_sovits_orchestrator.config import (
    TTS_SERVER_BASE_URL,
    V2PRO_API_TIMEOUT_S,
    V2PRO_EXTRACT_PATH,
    V2PRO_START_PATH,
    V2PRO_STOP_PATH,
)
from gpt_sovits_orchestrator.hubert.npz import hubert_npz_paths
from gpt_sovits_orchestrator.semantic.npz import semantic_npz_paths
from gpt_sovits_orchestrator.semantic.preprocess import hubert_to_semantic_input

TOKEN_MIN = 0
TOKEN_MAX = 1023


def _array_to_npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _check_server(base_url: str) -> None:
    try:
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            response = client.get("/api/hello")
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Cannot reach tts-server ({base_url}), ensure it is running: {exc}"
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


def _extract_semantic(
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


@task(name="prepare-semantic-inputs", log_prints=True)
def prepare_semantic_inputs(
    hubert_out_npz: Path,
    in_npz_path: Path,
) -> dict[str, np.ndarray]:
    """Transpose HuBERT output NPZ entries to semantic worker input NPZ."""
    hubert_out_npz = hubert_out_npz.resolve()
    in_npz_path = in_npz_path.resolve()
    in_npz_path.parent.mkdir(parents=True, exist_ok=True)

    if not hubert_out_npz.is_file():
        raise FileNotFoundError(f"HuBERT output NPZ not found: {hubert_out_npz}")

    hubert_data = np.load(hubert_out_npz)
    if not hubert_data.files:
        raise ValueError(f"HuBERT output NPZ is empty: {hubert_out_npz}")

    arrays: dict[str, np.ndarray] = {}
    keys = sorted(hubert_data.files)
    total = len(keys)

    for index, key in enumerate(keys, start=1):
        arrays[key] = hubert_to_semantic_input(hubert_data[key])
        print(f"OK   [{index}/{total}] {key} -> {arrays[key].shape}")

    np.savez(in_npz_path, **arrays)
    print(f"Saved semantic input NPZ: {in_npz_path} ({len(arrays)} arrays)")
    return arrays


@task(name="extract-semantic-tokens", log_prints=True)
def extract_semantic_tokens(
    inputs: dict[str, np.ndarray],
    out_npz_path: Path,
    hubert_out_npz: Path,
    *,
    base_url: str = TTS_SERVER_BASE_URL,
    preload: bool = True,
) -> Path:
    """Extract semantic tokens via tts-server v2pro API and save as NPZ."""
    out_npz_path = out_npz_path.resolve()
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)

    if not inputs:
        raise ValueError("Semantic inputs dict is empty")

    hubert_data = np.load(hubert_out_npz.resolve()) if hubert_out_npz.is_file() else None

    _check_server(base_url)
    outputs: dict[str, np.ndarray] = {}
    failures: list[str] = []

    with httpx.Client(base_url=base_url, timeout=V2PRO_API_TIMEOUT_S) as client:
        if preload:
            _api_start(client, V2PRO_START_PATH)
        try:
            for wav_key, array in sorted(inputs.items()):
                stem = Path(wav_key).stem
                try:
                    semantic = _extract_semantic(
                        stem,
                        _array_to_npy_bytes(array),
                        client=client,
                        extract_path=V2PRO_EXTRACT_PATH,
                    )
                    if semantic.dtype != np.int32:
                        semantic = semantic.astype(np.int32)
                    if semantic.ndim != 1:
                        raise ValueError(f"Expected shape (T_sem,), got {semantic.shape}")
                    if semantic.size == 0:
                        raise ValueError("Empty semantic output")
                    if semantic.min() < TOKEN_MIN or semantic.max() > TOKEN_MAX:
                        raise ValueError(
                            f"Token range [{semantic.min()}, {semantic.max()}] "
                            f"outside [{TOKEN_MIN}, {TOKEN_MAX}]"
                        )
                    if hubert_data is not None and wav_key in hubert_data.files:
                        expected_t = hubert_data[wav_key].shape[1] // 2
                        if semantic.shape[0] != expected_t:
                            raise ValueError(
                                f"T_sem={semantic.shape[0]}, expected hubert_T/2={expected_t}"
                            )
                    outputs[wav_key] = semantic
                    print(f"OK   {wav_key} -> semantic {semantic.shape}")
                except (RuntimeError, ValueError) as exc:
                    failures.append(wav_key)
                    print(f"FAIL {wav_key}: {exc}")
        finally:
            if preload:
                _api_stop(client, V2PRO_STOP_PATH)

    print(f"Extracted semantic tokens: {len(outputs)} ok, {len(failures)} failed")
    if failures:
        raise RuntimeError(f"Semantic extraction failed for: {', '.join(failures)}")

    np.savez(out_npz_path, **outputs)
    print(f"Saved semantic output NPZ: {out_npz_path} ({len(outputs)} arrays)")
    return out_npz_path


@task(name="semantic-from-hubert-out", log_prints=True)
def semantic_from_hubert_out(
    hubert_out_npz: Path,
    data_dir: Path,
    *,
    base_url: str = TTS_SERVER_BASE_URL,
    preload: bool = True,
) -> tuple[Path, Path]:
    """Transpose HuBERT output NPZ and extract semantic tokens to output NPZ."""
    hubert_out_npz = hubert_out_npz.resolve()
    data_dir = data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    in_npz_path, out_npz_path = semantic_npz_paths(hubert_out_npz, data_dir)
    inputs = prepare_semantic_inputs(hubert_out_npz, in_npz_path)
    extract_semantic_tokens(
        inputs,
        out_npz_path,
        hubert_out_npz,
        base_url=base_url,
        preload=preload,
    )
    return in_npz_path, out_npz_path


@task(name="semantic-from-zip", log_prints=True)
def semantic_from_zip(
    zip_path: Path,
    hubert_data_dir: Path,
    semantic_data_dir: Path,
    *,
    base_url: str = TTS_SERVER_BASE_URL,
    preload: bool = True,
) -> tuple[Path, Path]:
    """Resolve HuBERT output from slice ZIP stem and run semantic extraction."""
    _, hubert_out_npz = hubert_npz_paths(zip_path, hubert_data_dir)
    return semantic_from_hubert_out(
        hubert_out_npz,
        semantic_data_dir,
        base_url=base_url,
        preload=preload,
    )
