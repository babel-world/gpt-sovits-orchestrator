"""Source wav -> SoVITS s2 target audio (32 kHz int16, 32768 branch)."""

from __future__ import annotations

import io
import warnings
import wave
from pathlib import Path

import librosa
import numpy as np

from gpt_sovits_orchestrator.config import (
    INT16_SCALE,
    LOAD_SR,
    NORM_ALPHA,
    NORM_MAX,
    PEAK_LIMIT,
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


def _audio_to_wav32k_int16(audio: np.ndarray, peak: float) -> np.ndarray:
    scaled = (audio / peak * (NORM_MAX * NORM_ALPHA * INT16_SCALE)) + (
        (1 - NORM_ALPHA) * INT16_SCALE
    ) * audio
    return scaled.astype(np.int16, copy=False)


def wav_bytes_to_wav32k_int16(wav_bytes: bytes, label: str) -> np.ndarray | None:
    """Convert in-memory wav bytes to 32 kHz int16 target audio."""
    audio, _ = librosa.load(io.BytesIO(wav_bytes), sr=LOAD_SR, mono=True)
    peak = _check_peak(audio, label)
    if peak is None:
        return None
    return _audio_to_wav32k_int16(audio, peak)


def int16_array_to_wav_bytes(audio: np.ndarray, sample_rate: int = LOAD_SR) -> bytes:
    """Pack mono int16 PCM into a wav container."""
    if audio.dtype != np.int16:
        raise TypeError(f"Expected int16 audio, got {audio.dtype}")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(audio.tobytes())
    return buffer.getvalue()


def wav_to_wav32k_int16(wav_path: str | Path) -> np.ndarray | None:
    path = Path(wav_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return wav_bytes_to_wav32k_int16(path.read_bytes(), str(path))
