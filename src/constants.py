"""Central configuration values for the prediction framework."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Application settings
APP_TITLE = "Adaptive Financial Prediction Framework"
ASSET_NAME = "Bitcoin"
TRADING_PAIR = "BTC/USD"
TIMEFRAME = "Daily"

# Model settings
CANDIDATE_LOOKBACKS = [5, 10, 20, 30, 60]
DEFAULT_LOOKBACK = 20
MIN_CUSTOM_LOOKBACK = 1
MAX_CUSTOM_LOOKBACK = 90

ROLLING_EVALUATION_WINDOW = 30
MINIMUM_TRAINING_ROWS = 100

RANDOM_STATE = 42
N_ESTIMATORS = 200

# Dataset columns
DATE_COLUMN = "date"
OPEN_COLUMN = "open"
HIGH_COLUMN = "high"
LOW_COLUMN = "low"
CLOSE_COLUMN = "close"
VOLUME_COLUMN = "volume"
TARGET_COLUMN = "target"

REQUIRED_COLUMNS = [
    DATE_COLUMN,
    OPEN_COLUMN,
    HIGH_COLUMN,
    LOW_COLUMN,
    CLOSE_COLUMN,
    VOLUME_COLUMN,
]

PREDICTION_LABELS = {
    0: "Bearish",
    1: "Bullish",
}