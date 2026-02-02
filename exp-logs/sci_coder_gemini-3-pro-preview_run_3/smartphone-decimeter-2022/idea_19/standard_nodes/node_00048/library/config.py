import os
import numpy as np

# =============================================================================
# 1. File Paths & Directories
# =============================================================================
# Root directory for input data (Read-Only)
INPUT_DIR = "./input"

# Subdirectories
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

# Metadata paths (Pre-generated)
METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Sample submission
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Working directory for caching intermediate files (Write Allowed)
# We use a specific subdirectory for this idea to avoid conflicts
WORKING_DIR = "./working/idea_19"
os.makedirs(WORKING_DIR, exist_ok=True)

# Output directory for final submission
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# 2. Physical Constants
# =============================================================================
SPEED_OF_LIGHT = 299_792_458.0  # m/s
OMEGA_EARTH = 7.2921151467e-5  # rad/s (Earth rotation rate)

# GNSS Frequencies (Hz)
# L1 Band Center (GPS L1, GAL E1, QZSS L1, BDS B1C)
GPS_L1_FREQ = 1575.42e6
# L5 Band Center (GPS L5, GAL E5a, QZSS L5, BDS B2a)
GPS_L5_FREQ = 1176.45e6

# Frequency thresholds for band splitting logic
# Signals within these ranges will be categorized as L1 or L5 respectively
FREQ_TOLERANCE = 10.0e6  # +/- 10 MHz
L1_BAND_MIN = 1550e6
L1_BAND_MAX = 1615e6  # Includes GLONASS G1 (~1602)
L5_BAND_MIN = 1160e6
L5_BAND_MAX = 1190e6

# =============================================================================
# 3. Data Processing Configuration
# =============================================================================
# Random Seed for reproducibility
SEED = 42

# Debugging flags
DEBUG = False  # Set to True to run on a smaller subset of data
DEBUG_SAMPLE_SIZE = 1000  # Number of rows to process if DEBUG is True

# Caching control
LOAD_CACHED_DATA = True  # If True, attempts to load .parquet files from WORKING_DIR

# Coordinate System Reference (WGS84 Ellipsoid)
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)

# =============================================================================
# 4. Model Hyperparameters (LightGBM)
# =============================================================================
# Parameters for the Split-Band Projected Boosting model
LGBM_PARAMS = {
    "objective": "mae",  # Mean Absolute Error (L1 loss) for robustness to outliers
    "boosting_type": "gbdt",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 128,
    "max_depth": 10,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": -1,
}

# Early stopping rounds
EARLY_STOPPING_ROUNDS = 100

# =============================================================================
# 5. Graph Optimization Parameters
# =============================================================================
# Weights for the Factor Graph cost function
# J(x) = Sum( Huber(x - x_anchor) ) + Sum( w * ||(x_t - x_{t-1}) - dx_kin||^2 )

# Anchor term parameters
HUBER_DELTA = (
    5.0  # Threshold for Huber loss (meters) - transitions from quadratic to linear
)

# Kinematic edge weights (Inverse variance)
# High weight = trust kinematics more
WEIGHT_TDCP = 10.0  # Weight when Carrier Phase (TDCP) is valid
WEIGHT_DOPPLER = 1.0  # Weight when falling back to Doppler velocity

# RANSAC parameters for TDCP estimation
RANSAC_MIN_SAMPLES = 4
RANSAC_RESIDUAL_THRESHOLD = 0.05  # meters/second
RANSAC_MAX_TRIALS = 100

# Carrier Phase Cycle Slip Detection
CYCLE_SLIP_THRESHOLD = 1.0  # meters (Unexplained jump in phase)
