from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EXAMPLES_DIR = PROJECT_ROOT / "examples"

DEFAULT_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 320
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 5
DEFAULT_LR = 2e-5
DEFAULT_RANDOM_STATE = 42
DEFAULT_VAL_SIZE = 0.2
DEFAULT_PATIENCE = 2

MIN_TEXT_LENGTH = 20

DATASET_CANDIDATES = [
    "hendrycks/competition_math",
    "nlile/hendrycks-MATH-benchmark",
]
