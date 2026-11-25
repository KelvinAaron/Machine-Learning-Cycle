import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Overridable so Docker volumes can be shared between containers.
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))

RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
REGISTRY_PATH = MODELS_DIR / "registry.json"
FEATURE_CACHE_PATH = DATA_DIR / "feature_cache.json"
RETRAIN_STATE_PATH = MODELS_DIR / "retrain_state.json"
RETRAIN_LOCK_PATH = MODELS_DIR / "retrain.lock"
RETRAIN_LOCK_TIMEOUT = 6 * 3600

DATASET_URLS = [
    "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip",
]

IMG_SIZE = (128, 128)
INPUT_SHAPE = (128, 128, 3)
BATCH_SIZE = 32
SEED = 42

# Index in this list is the label, so parasitized is the positive class (1).
CLASS_NAMES = ["uninfected", "parasitized"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

# Starting values for the colour features; adjust after looking at the EDA sample grid.
BACKGROUND_THRESHOLD = 15
STAIN_GREEN_THRESHOLD = 100

RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "50"))
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "true").lower() == "true"
OLD_SAMPLE_SIZE = int(os.getenv("OLD_SAMPLE_SIZE", "4000"))
PROMOTION_TOLERANCE = 0.005

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
