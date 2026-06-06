from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = PROJECT_ROOT / ".local"
DATA_01_DIR = LOCAL_DIR / "data_01"
DATA_02_DIR = LOCAL_DIR / "data_02"

ASR_SERVER_BASE_URL = "http://127.0.0.1:19031"
SLICE_API_PATH = "/api/audio/slice"
