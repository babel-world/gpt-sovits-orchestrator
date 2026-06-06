"""Source wav -> HuBERT extract input .npy (RVC/SoVITS 1145.14 branch)."""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import librosa
import numpy as np

from gpt_sovits_orchestrator.config import (
    LOAD_SR,
    NORM_ALPHA,
    NORM_MAX,
    NORM_SCALE,
    PEAK_LIMIT,
    TARGET_SR,
)


def _check_peak(audio: np.ndarray, label: str) -> float | None:
    peak = float(np.abs(audio).max())
    if peak > PEAK_LIMIT:
        warnings.warn(f"跳过: {label} 峰值异常 ({peak:.2f} > {PEAK_LIMIT})")
        return None
    if peak == 0:
        warnings.warn(f"跳过: {label} 为纯静音")
        return None
    return peak


def _audio_to_hubert_input(audio: np.ndarray, peak: float) -> np.ndarray:
    scaled = (audio / peak * (NORM_MAX * NORM_ALPHA * NORM_SCALE)) + (
        (1 - NORM_ALPHA) * NORM_SCALE
    ) * audio
    resampled = librosa.resample(scaled, orig_sr=LOAD_SR, target_sr=TARGET_SR)
    return resampled.astype(np.float32, copy=False)


def load_and_check_audio(wav_path: str | Path) -> tuple[np.ndarray, float] | None:
    """Load 32 kHz audio and run peak checks. Returns None when skipped."""
    path = Path(wav_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, _ = librosa.load(path, sr=LOAD_SR, mono=True)
    peak = _check_peak(audio, str(path))
    if peak is None:
        return None
    return audio, peak


def wav_bytes_to_hubert_input(wav_bytes: bytes, label: str) -> np.ndarray | None:
    """Convert in-memory wav bytes to extract-ready 1D float32 array."""
    audio, _ = librosa.load(io.BytesIO(wav_bytes), sr=LOAD_SR, mono=True)
    peak = _check_peak(audio, label)
    if peak is None:
        return None
    return _audio_to_hubert_input(audio, peak)


def wav_to_hubert_input(wav_path: str | Path) -> np.ndarray | None:
    """Convert source wav to extract-ready 1D float32 array (T,)."""
    loaded = load_and_check_audio(wav_path)
    if loaded is None:
        return None
    audio, peak = loaded
    return _audio_to_hubert_input(audio, peak)


def save_hubert_input_npy(wav_path: str | Path, out_path: str | Path) -> Path:
    """Convert a single wav file and save as .npy."""
    arr = wav_to_hubert_input(wav_path)
    if arr is None:
        raise RuntimeError(f"Preprocess skipped: {wav_path}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    return out_path
