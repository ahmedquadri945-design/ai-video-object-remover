import os
import shutil
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
TEMP_DIR = DATA_DIR / "temp"

# Ensure runtime directories exist
for directory in [DATA_DIR, UPLOADS_DIR, PROCESSED_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File upload settings
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# External tools detection
FFMPEG_PATH = shutil.which("ffmpeg") or os.environ.get("FFMPEG_PATH")
if not FFMPEG_PATH:
    try:
        import imageio_ffmpeg
        FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        FFMPEG_PATH = None

FFPROBE_PATH = shutil.which("ffprobe") or os.environ.get("FFPROBE_PATH")

# Server settings
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))
CORS_ORIGINS = ["*"]
