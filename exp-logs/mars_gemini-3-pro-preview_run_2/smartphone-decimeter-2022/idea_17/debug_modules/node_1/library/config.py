import os

# ---------------------------------------------------------
# 1. File Paths and Directories
# ---------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_17")
SUBMISSION_DIR = "./submission"

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# ---------------------------------------------------------
# 2. Data Parameters
# ---------------------------------------------------------
# Random Seed for Reproducibility
SEED = 42

# Windowing
# Size of the sliding window (number of epochs)
WINDOW_SIZE = 15

# Feature Definitions
# Kinematic features: Time-series input for the 1D-CNN
# These are computed per epoch.
# - *_rel_m: Position relative to the window center (meters)
# - *_diff_m: Velocity/Delta between epochs (meters)
# - MeanCn0: Signal strength
KINEMATIC_FEATURES = [
    "wls_lat_rel_m",
    "wls_lon_rel_m",
    "wls_alt_rel_m",
    "wls_lat_diff_m",
    "wls_lon_diff_m",
    "wls_alt_diff_m",
    "MeanCn0",
]

# Sky Context features: Aggregated statistics for the MLP
# These are aggregated over the entire window.
# We will compute mean and std for these.
SKY_FEATURES = [
    "SvElevationDegrees",
    "SvAzimuthDegrees",
    "Cn0DbHz",  # Raw Cn0 from individual satellites to aggregate
]

# Target Variables (Residuals in meters)
TARGET_COLUMNS = ["dLat_m", "dLon_m"]

# ---------------------------------------------------------
# 3. Training Hyperparameters
# ---------------------------------------------------------
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
NUM_WORKERS = 4  # For DataLoader

# Model Architecture Config
CNN_HIDDEN_DIM = 64
CNN_KERNEL_SIZE = 3
CNN_LAYERS = 3
CNN_DROPOUT = 0.1

MLP_HIDDEN_DIM = 128
MLP_DROPOUT = 0.1

# ---------------------------------------------------------
# 4. Caching Configuration
# ---------------------------------------------------------
# Filenames for cached numpy arrays
TRAIN_CACHE_FILES = {
    "X_kin": os.path.join(CACHE_DIR, "train_X_kinematic.npy"),
    "X_sky": os.path.join(CACHE_DIR, "train_X_sky.npy"),
    "y": os.path.join(CACHE_DIR, "train_y.npy"),
    "meta": os.path.join(CACHE_DIR, "train_meta.parquet"),
}

VAL_CACHE_FILES = {
    "X_kin": os.path.join(CACHE_DIR, "val_X_kinematic.npy"),
    "X_sky": os.path.join(CACHE_DIR, "val_X_sky.npy"),
    "y": os.path.join(CACHE_DIR, "val_y.npy"),
    "meta": os.path.join(CACHE_DIR, "val_meta.parquet"),
}

TEST_CACHE_FILES = {
    "X_kin": os.path.join(CACHE_DIR, "test_X_kinematic.npy"),
    "X_sky": os.path.join(CACHE_DIR, "test_X_sky.npy"),
    "meta": os.path.join(CACHE_DIR, "test_meta.parquet"),
}

SCALER_PATH = os.path.join(CACHE_DIR, "scaler.joblib")
MODEL_PATH = os.path.join(CACHE_DIR, "best_model.pth")
