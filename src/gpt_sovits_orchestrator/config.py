from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = PROJECT_ROOT / ".local"
DATA_01_DIR = LOCAL_DIR / "data_01"
DATA_02_DIR = LOCAL_DIR / "data_02"
DATA_03_DIR = LOCAL_DIR / "data_03"
DATA_04_DIR = LOCAL_DIR / "data_04"

ASR_SERVER_BASE_URL = "http://127.0.0.1:19031"
SLICE_API_PATH = "/api/audio/slice"
TRANSCRIBE_API_PATH = "/api/transcribe"

NLP_SERVER_BASE_URL = "http://127.0.0.1:19032"
G2P_JA_API_PATH = "/api/g2p/ja"
G2P_MODE = "prosody"

MANIFEST_FIELDS = ("filename", "speaker", "language", "text", "probability")
G2P_COLUMNS = ("norm_text", "phones", "phone_count", "word2ph", "status", "error")
G2P_OUTPUT_COLUMNS = MANIFEST_FIELDS + G2P_COLUMNS

PRIVACY_ALLOWED_LANGUAGE = "ja"
PRIVACY_MIN_PROBABILITY = 0.95
