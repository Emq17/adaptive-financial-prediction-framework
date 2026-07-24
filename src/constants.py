"""Central configuration values for the prediction framework."""

# Application settings
APP_TITLE = "Adaptive Financial Prediction Framework"

# Model settings
CANDIDATE_LOOKBACKS = [5, 10, 20, 30, 60]
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
