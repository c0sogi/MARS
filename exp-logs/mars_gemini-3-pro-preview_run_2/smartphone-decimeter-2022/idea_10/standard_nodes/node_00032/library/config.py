import os
import torch

# ==========================================
# Global Random Seed
# ==========================================
SEED = 42

# ==========================================
# File Paths and Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_10"
SUBMISSION_DIR = "./submission"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Cache Files (Parquet format)
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_data_cache.parquet")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_data_cache.parquet")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_data_cache.parquet")

# Scaler Path
SCALER_PATH = os.path.join(WORKING_DIR, "scaler.json")

# Model Checkpoint
MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# Submission Output
SUBMISSION_OUTPUT_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Processing Hyperparameters
# ==========================================
# Size of the sliding window (number of epochs)
WINDOW_SIZE = 15

# Scaling factor for latitude degrees to meters (approximate)
LAT_DEG_TO_METERS = 111320.0

# ==========================================
# Feature Definitions
# ==========================================
# Features for the Trajectory Stream (Time-series input)
# These will be computed per epoch in the window
TRAJECTORY_FEATURES = [
    "rel_lat_m",  # Relative Latitude in meters (centered on window)
    "rel_lon_m",  # Relative Longitude in meters (centered on window)
    "rel_alt_m",  # Relative Altitude in meters (centered on window)
    "vel_lat_m",  # Velocity Latitude (diff)
    "vel_lon_m",  # Velocity Longitude (diff)
    "vel_alt_m",  # Velocity Altitude (diff)
    "raw_cn0",  # Signal strength
    "raw_uncertainty",  # ReceivedSvTimeUncertaintyNanos or similar
]

# Features for the Sky Context Stream (Aggregated input)
# These will be aggregated statistics over the window
SKY_FEATURES = [
    "mean_elev",  # Mean Satellite Elevation
    "std_elev",  # Standard Deviation of Satellite Elevation
    "mean_azim",  # Mean Satellite Azimuth
    "std_azim",  # Standard Deviation of Satellite Azimuth
    "mean_cn0_sky",  # Mean Signal Strength (Sky view)
    "sat_count",  # Number of satellites
]

# Target variables (Residuals in meters)
TARGET_FEATURES = [
    "d_lat_m",  # Ground Truth Lat - Baseline Lat (in meters)
    "d_lon_m",  # Ground Truth Lon - Baseline Lon (in meters)
]

# ==========================================
# Model Architecture Hyperparameters
# ==========================================
# Trajectory Stream (1D CNN)
CNN_CHANNELS = 64
CNN_KERNEL_SIZE = 3
CNN_LAYERS = 3
CNN_DROPOUT = 0.2

# Sky Stream (MLP)
SKY_HIDDEN_DIM = 32
SKY_DROPOUT = 0.1

# Fusion Head
FUSION_HIDDEN_DIM = 128
FUSION_DROPOUT = 0.2

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
PATIENCE = 10  # Early stopping patience
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Debugging: Set to a small number to limit dataset size during development
# Set to None for full training
DEBUG_SAMPLE_SIZE = None
