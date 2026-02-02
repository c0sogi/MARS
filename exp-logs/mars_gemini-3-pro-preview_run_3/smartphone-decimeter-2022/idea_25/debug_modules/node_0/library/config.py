import os
import numpy as np

# =============================================================================
# 1. File System Configuration
# =============================================================================

# Root directories
INPUT_DIR = "./input"
# Cache directory for this specific idea (Idea 25)
OUTPUT_DIR = "./working/idea_25"
SUBMISSION_DIR = "./submission"
METADATA_DIR = "./metadata"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata file paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# =============================================================================
# 2. Physical Constants (WGS84)
# =============================================================================
# Used for ECEF <-> LLA <-> ENU transformations
WGS84_A = 6378137.0  # Semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563  # Flattening factor
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis
WGS84_E2 = (WGS84_A**2 - WGS84_B**2) / (WGS84_A**2)  # First eccentricity squared

# =============================================================================
# 3. Data Processing Configuration
# =============================================================================

# Random seed for reproducibility
SEED = 42

# Features to be used by the ML model
# Explicitly including Receiver State features as per Idea 25
# Explicitly excluding Doppler features
ML_FEATURES = [
    # Signal Quality
    "Cn0DbHz_mean",
    "Cn0DbHz_std",
    # Geometric Projection Features (Unified Force)
    # These represent the aggregate error vector implied by pseudoranges
    "Unified_Force_E",
    "Unified_Force_N",
    "Unified_Force_U",
    # Geometry Stiffness (Dilution of Precision proxies)
    "HDOP",
    "VDOP",
    "PDOP",
    # Receiver State (Crucial for correcting internal bias)
    "BiasNanos",
    "BiasUncertaintyNanos",
    "DriftNanosPerSecond",
    "DriftUncertaintyNanosPerSecond",
    # Satellite Stats
    "SvElevationDegrees_mean",
    "SvAzimuthDegrees_mean",
    "SatCount",
]

# Target definitions
# We predict ENU residuals relative to the WLS baseline
# Altitude correction is applied during target generation
TARGET_COLS = ["res_E", "res_N"]

# =============================================================================
# 4. Model Hyperparameters (LightGBM)
# =============================================================================

LGBM_PARAMS = {
    "objective": "mae",  # L1 loss to be robust against GNSS outliers
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# Training settings
N_FOLDS = 5  # Number of folds for GroupKFold
EARLY_STOPPING_ROUNDS = 100

# =============================================================================
# 5. Graph Optimization Parameters
# =============================================================================

# Weight for the smoothness/kinematic term in the graph optimizer
# Higher lambda -> trust odometry (TDCP) more
# Lower lambda -> trust ML anchors more
OPT_LAMBDA = 5.0

# Huber loss delta for the anchor term
# Residuals larger than this (in meters) will have linear penalty instead of quadratic
# This makes the optimizer robust to large ML prediction errors
HUBER_DELTA = 5.0

# RANSAC parameters for TDCP velocity estimation
RANSAC_THRESHOLD = 0.5  # meters/second
RANSAC_MIN_SAMPLES = 4
