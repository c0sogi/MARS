import os
import torch
import numpy as np
import random

# ==========================================
# File Paths and Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_16"
SUBMISSION_DIR = "./submission"

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# ==========================================
# Data Parameters
# ==========================================
# Window size in epochs (seconds).
# A window of 15 seconds captures local motion context.
WINDOW_SIZE = 15
# The index of the target timestamp within the window (center)
WINDOW_CENTER_IDX = WINDOW_SIZE // 2

# Columns to load from raw GNSS files (device_gnss.csv)
GNSS_COLS = [
    "utcTimeMillis",
    "WlsPositionXEcefMeters",
    "WlsPositionYEcefMeters",
    "WlsPositionZEcefMeters",
    "SvElevationDegrees",
    "SvAzimuthDegrees",
    "Cn0DbHz",
    "RawPseudorangeUncertaintyMeters",
    "ReceivedSvTimeUncertaintyNanos",
]

# Columns to load from raw IMU files (device_imu.csv)
# We focus on Uncalibrated Accelerometer for dynamics
IMU_COLS = [
    "utcTimeMillis",
    "MeasurementX",
    "MeasurementY",
    "MeasurementZ",
]

# Columns to load from Ground Truth (ground_truth.csv)
GT_COLS = [
    "UnixTimeMillis",
    "LatitudeDegrees",
    "LongitudeDegrees",
    "AltitudeMeters",
]

# Feature definitions for the model inputs
# 1. Trajectory Features (per timestep in window)
# These features are computed during preprocessing and represent the state at each step relative to the window center.
TRAJ_FEATURES = [
    "rel_pos_x",
    "rel_pos_y",
    "rel_pos_z",  # Position in meters relative to window center WLS
    "vel_x",
    "vel_y",
    "vel_z",  # Velocity derived from WLS differences
    "acc_x",
    "acc_y",
    "acc_z",  # Acceleration from IMU (aligned to GNSS epochs)
    "mean_cn0",  # Mean Carrier-to-Noise density per epoch
    "mean_pr_unc",  # Mean Pseudorange Uncertainty per epoch
    "mean_sv_time_unc",  # Mean Received SV Time Uncertainty per epoch
]

# 2. Sky-State Features (aggregated over the entire window)
# These provide environmental context (e.g., open sky vs urban canyon)
SKY_FEATURES = [
    "mean_elev",
    "std_elev",  # Satellite Elevation stats
    "mean_azim",
    "std_azim",  # Satellite Azimuth stats
    "mean_cn0_sky",
    "std_cn0_sky",  # Signal strength stats over window
    "sat_count_mean",  # Average number of satellites visible
]

NUM_TRAJ_FEATURES = len(TRAJ_FEATURES)
NUM_SKY_FEATURES = len(SKY_FEATURES)

# Input dimensions for the model
# Trajectory input is flattened: Window Size * Features per step
INPUT_DIM_TRAJ = WINDOW_SIZE * NUM_TRAJ_FEATURES
INPUT_DIM_SKY = NUM_SKY_FEATURES

# Output dimension: 2 (Delta East, Delta North in meters)
OUTPUT_DIM = 2

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
EPOCHS = 50
PATIENCE = 10  # For Early Stopping
RANDOM_STATE = 42
NUM_WORKERS = 4  # For DataLoader

# Model Architecture
# Dimensions for the MLP layers
TRAJ_HIDDEN_DIMS = [512, 256, 128]
SKY_HIDDEN_DIMS = [64, 32]
FUSION_HIDDEN_DIMS = [128, 64]
DROPOUT = 0.2

# ==========================================
# Constants
# ==========================================
# Approximate meters per degree latitude (WGS84)
LAT_METERS_PER_DEGREE = 111320.0

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=RANDOM_STATE):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to RANDOM_STATE.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
