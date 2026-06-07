import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

LOCAL_DIR = PROJECT_ROOT / ".local"
DATA_01_DIR = LOCAL_DIR / "data_01"
DATA_02_DIR = LOCAL_DIR / "data_02"
DATA_03_DIR = LOCAL_DIR / "data_03"
DATA_04_DIR = LOCAL_DIR / "data_04"
DATA_05_DIR = LOCAL_DIR / "data_05"
DATA_06_DIR = LOCAL_DIR / "data_06"
DATA_07_DIR = LOCAL_DIR / "data_07"
DATA_08_DIR = LOCAL_DIR / "data_08"
DATA_09_DIR = LOCAL_DIR / "data_09"


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    return float(raw) if raw is not None else default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw is not None else default


ASR_SERVER_BASE_URL = _env_str("ASR_SERVER_BASE_URL", "http://127.0.0.1:19031")
TTS_SERVER_BASE_URL = _env_str("TTS_SERVER_BASE_URL", "http://127.0.0.1:19033")
NLP_SERVER_BASE_URL = _env_str("NLP_SERVER_BASE_URL", "http://127.0.0.1:19032")

SLICE_API_PATH = "/api/audio/slice"
TRANSCRIBE_API_PATH = "/api/transcribe"
HUBERT_EXTRACT_PATH = "/api/features/chinese-hubert-base"
HUBERT_START_PATH = "/api/features/chinese-hubert-base/start"
HUBERT_STOP_PATH = "/api/features/chinese-hubert-base/stop"
SV_EXTRACT_PATH = "/api/features/speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common"
SV_START_PATH = "/api/features/speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common/start"
SV_STOP_PATH = "/api/features/speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common/stop"
V2PRO_EXTRACT_PATH = "/api/features/v2pro"
V2PRO_START_PATH = "/api/features/v2pro/start"
V2PRO_STOP_PATH = "/api/features/v2pro/stop"

SLICE_API_TIMEOUT_S = _env_float("SLICE_API_TIMEOUT_S", 600.0)
TRANSCRIBE_API_TIMEOUT_S = _env_float("TRANSCRIBE_API_TIMEOUT_S", 600.0)
G2P_API_TIMEOUT_S = _env_float("G2P_API_TIMEOUT_S", 60.0)
HUBERT_API_TIMEOUT_S = _env_float("HUBERT_API_TIMEOUT_S", 120.0)
SV_API_TIMEOUT_S = _env_float("SV_API_TIMEOUT_S", 120.0)
V2PRO_API_TIMEOUT_S = _env_float("V2PRO_API_TIMEOUT_S", 120.0)
SERVER_HEALTH_CHECK_TIMEOUT_S = _env_float("SERVER_HEALTH_CHECK_TIMEOUT_S", 10.0)

# Passed to asr-server POST /api/audio/slice (Form fields)
SLICE_THRESHOLD_DB = _env_float("SLICE_THRESHOLD_DB", -40.0)
SLICE_MIN_LENGTH_MS = _env_int("SLICE_MIN_LENGTH_MS", 5000)
SLICE_MIN_INTERVAL_MS = _env_int("SLICE_MIN_INTERVAL_MS", 300)
SLICE_HOP_SIZE_MS = _env_int("SLICE_HOP_SIZE_MS", 20)
SLICE_MAX_SIL_KEPT_MS = _env_int("SLICE_MAX_SIL_KEPT_MS", 5000)

LOAD_SR = _env_int("LOAD_SR", 32_000)
TARGET_SR = _env_int("TARGET_SR", 16_000)
PEAK_LIMIT = _env_float("PEAK_LIMIT", 2.2)
PEAK_NORM = _env_float("PEAK_NORM", 0.95)
NORM_MAX = _env_float("NORM_MAX", 0.95)
NORM_ALPHA = _env_float("NORM_ALPHA", 0.5)
NORM_SCALE = _env_float("NORM_SCALE", 1145.14)
INT16_SCALE = _env_float("INT16_SCALE", 32768.0)

G2P_JA_API_PATH = "/api/g2p/ja"
G2P_MODE = _env_str("G2P_MODE", "prosody")

MANIFEST_FIELDS = ("filename", "speaker", "language", "text", "probability")
G2P_COLUMNS = ("norm_text", "phones", "phone_count", "word2ph", "status", "error")
G2P_OUTPUT_COLUMNS = MANIFEST_FIELDS + G2P_COLUMNS

PRIVACY_ALLOWED_LANGUAGE = _env_str("PRIVACY_ALLOWED_LANGUAGE", "ja")
PRIVACY_MIN_PROBABILITY = _env_float("PRIVACY_MIN_PROBABILITY", 0.95)

ASSEMBLE_AUDIT_COLUMNS = ("semantic_count", "hubert_T", "sv_dim", "wav32k_bytes")
ASSEMBLE_ELIGIBLE_COLUMNS = ("eligible_gpt", "eligible_sovits")
ASSEMBLE_OUTPUT_COLUMNS = G2P_OUTPUT_COLUMNS + ASSEMBLE_AUDIT_COLUMNS + ASSEMBLE_ELIGIBLE_COLUMNS
