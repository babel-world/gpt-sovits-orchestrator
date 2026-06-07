"""Source wav -> SV worker input arrays (16 kHz float32, peak norm, no 1145.14 scale)."""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import librosa
import numpy as np

from gpt_sovits_orchestrator.config import LOAD_SR, PEAK_LIMIT, PEAK_NORM, TARGET_SR


def _check_peak(audio: np.ndarray, label: str) -> float | None:
    peak = float(np.abs(audio).max())
    if peak > PEAK_LIMIT:
        warnings.warn(f"跳过: {label} 峰值异常 ({peak:.2f} > {PEAK_LIMIT})")
        return None
    if peak == 0:
        warnings.warn(f"跳过: {label} 为纯静音")
        return None
    return peak


def _audio_to_sv_input(audio: np.ndarray, peak: float) -> np.ndarray:
    normalized = audio / peak * PEAK_NORM
    resampled = librosa.resample(normalized, orig_sr=LOAD_SR, target_sr=TARGET_SR)
    return resampled.astype(np.float32, copy=False)


def load_and_check_audio(wav_path: str | Path) -> tuple[np.ndarray, float] | None:
    path = Path(wav_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, _ = librosa.load(path, sr=LOAD_SR, mono=True)
    peak = _check_peak(audio, str(path))
    if peak is None:
        return None
    return audio, peak


def wav_bytes_to_sv_input(wav_bytes: bytes, label: str) -> np.ndarray | None:
    audio, _ = librosa.load(io.BytesIO(wav_bytes), sr=LOAD_SR, mono=True)
    peak = _check_peak(audio, label)
    if peak is None:
        return None
    return _audio_to_sv_input(audio, peak)


def wav_to_sv_input(wav_path: str | Path) -> np.ndarray | None:
    loaded = load_and_check_audio(wav_path)
    if loaded is None:
        return None
    audio, peak = loaded
    return _audio_to_sv_input(audio, peak)
