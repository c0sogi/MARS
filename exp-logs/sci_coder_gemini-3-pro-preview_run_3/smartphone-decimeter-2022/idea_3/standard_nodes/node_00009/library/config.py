import os

# -----------------------------------------------------------------------------
# Directory Configurations
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_3_fixed"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# File Paths
# -----------------------------------------------------------------------------
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache file paths for processed sequences
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

# -----------------------------------------------------------------------------
# Data Processing Parameters
# -----------------------------------------------------------------------------
# Raw columns to load from GNSS files
GNSS_RAW_COLS = [
    "utcTimeMillis",
    "Cn0DbHz",
    "SvElevationDegrees",
    "Svid",
]

# Raw columns to load from IMU files
IMU_RAW_COLS = [
    "utcTimeMillis",
    "MeasurementX",
    "MeasurementY",
    "MeasurementZ",
    "MessageType",  # To filter for UncalAccel
]

# Final feature names after aggregation
FEATURE_NAMES = [
    "Cn0DbHz_mean",
    "Cn0DbHz_std",
    "Cn0DbHz_max",
    "Svid_count",
    "SvElevationDegrees_mean",
    "Accel_mag_mean",
    "Accel_mag_std",
]

INPUT_SIZE = len(FEATURE_NAMES)
OUTPUT_SIZE = 2  # Latitude, Longitude corrections

# Sequence parameters
# Trips can be up to ~1 hour (3600s), but many are shorter.
# We set a max length to handle memory constraints and batching.
MAX_SEQUENCE_LENGTH = 2048

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = True

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
SEED = 42
NUM_WORKERS = 4  # For data loading
