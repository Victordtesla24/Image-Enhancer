import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Temp directory for file uploads
UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Supported file types
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Maximum file size (in MB)
MAX_FILE_SIZE = 10

# Default enhancement settings
DEFAULT_TARGET_WIDTH = 5120
