from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = PROJECT_ROOT / ".local"
DATA_01_DIR = LOCAL_DIR / "data_01"
DATA_02_DIR = LOCAL_DIR / "data_02"
DATA_03_DIR = LOCAL_DIR / "data_03"
DATA_04_DIR = LOCAL_DIR / "data_04"
DATA_05_DIR = LOCAL_DIR / "data_05"
DATA_06_DIR = LOCAL_DIR / "data_06"
DATA_07_DIR = LOCAL_DIR / "data_07"
DATA_08_DIR = LOCAL_DIR / "data_08"

ASR_SERVER_BASE_URL = "http://127.0.0.1:19031"
TTS_SERVER_BASE_URL = "http://127.0.0.1:19033"
SLICE_API_PATH = "/api/audio/slice"
TRANSCRIBE_API_PATH = "/api/transcribe"
HUBERT_EXTRACT_PATH = "/api/features/chinese-hubert-base"
HUBERT_START_PATH = "/api/features/chinese-hubert-base/start"
HUBERT_STOP_PATH = "/api/features/chinese-hubert-base/stop"
HUBERT_API_TIMEOUT_S = 120.0

SV_EXTRACT_PATH = "/api/features/speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common"
SV_START_PATH = "/api/features/speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common/start"
SV_STOP_PATH = "/api/features/speech-eres2netv2w24s4ep4-sv-zh-cn-16k-common/stop"
SV_API_TIMEOUT_S = 120.0

V2PRO_EXTRACT_PATH = "/api/features/v2pro"
V2PRO_START_PATH = "/api/features/v2pro/start"
V2PRO_STOP_PATH = "/api/features/v2pro/stop"
V2PRO_API_TIMEOUT_S = 120.0

LOAD_SR = 32_000
TARGET_SR = 16_000
PEAK_LIMIT = 2.2
PEAK_NORM = 0.95
NORM_MAX = 0.95
NORM_ALPHA = 0.5
NORM_SCALE = 1145.14
INT16_SCALE = 32768.0

NLP_SERVER_BASE_URL = "http://127.0.0.1:19032"
G2P_JA_API_PATH = "/api/g2p/ja"
G2P_MODE = "prosody"

MANIFEST_FIELDS = ("filename", "speaker", "language", "text", "probability")
G2P_COLUMNS = ("norm_text", "phones", "phone_count", "word2ph", "status", "error")
G2P_OUTPUT_COLUMNS = MANIFEST_FIELDS + G2P_COLUMNS

PRIVACY_ALLOWED_LANGUAGE = "ja"
PRIVACY_MIN_PROBABILITY = 0.95
