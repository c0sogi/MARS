import os

# ==========================================
# 1. File Paths & Directories
# ==========================================
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
METADATA_DIR = "./metadata"

# Output / Working Directory
# Strategy: State-Uncertainty Weighted Graph Fusion with Altitude-Corrected Geometric Boosting
IDEA_NAME = "idea_26"
WORKING_DIR = os.path.join("./working", IDEA_NAME)
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Submission
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# 2. Physical Constants (WGS84)
# ==========================================
# Semi-major axis
WGS84_A = 6378137.0
# Flattening
WGS84_F = 1.0 / 298.257223563
# Semi-minor axis
WGS84_B = WGS84_A * (1.0 - WGS84_F)
# First eccentricity squared
WGS84_E2 = WGS84_F * (2 - WGS84_F)
# Rotation rate of Earth (rad/s)
OMEGA_E = 7.2921151467e-5
# Speed of light (m/s)
CLLIGHT = 299792458.0

# ==========================================
# 3. Feature & Target Definitions
# ==========================================
# Features used for the LightGBM Model
# Excludes Doppler-derived features (PseudorangeRate) to avoid noise in static estimation
ML_FEATURES = [
    # Unified Geometric Force (Gradient of position error)
    "force_x",
    "force_y",
    "force_z",
    # Geometry Stiffness (Diagonal elements of DOP matrix)
    "stiffness_x",
    "stiffness_y",
    "stiffness_z",
    # Receiver Clock State (Critical for weighting)
    "BiasNanos",
    "BiasUncertaintyNanos",
    "DriftNanosPerSecond",
    "DriftUncertaintyNanosPerSecond",
    # Signal Quality Aggregates
    "Cn0DbHz_mean",
    "sv_count",
]

# Targets for the ML Model
# Altitude-Corrected ENU Residuals:
# Ground Truth projected to WLS Altitude, then differenced from WLS Baseline
TARGET_E = "dE_ac"
TARGET_N = "dN_ac"

# ==========================================
# 4. Model Hyperparameters (LightGBM)
# ==========================================
LGBM_PARAMS = {
    "objective": "mae",  # L1 Loss for robustness to outliers
    "n_estimators": 2000,  # Sufficient capacity
    "learning_rate": 0.05,  # Conservative learning rate
    "num_leaves": 128,  # Model complexity
    "colsample_bytree": 0.8,  # Feature subsampling
    "subsample": 0.7,  # Data subsampling
    "reg_alpha": 0.1,  # L1 regularization
    "reg_lambda": 0.1,  # L2 regularization
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# Training Configuration
TRAIN_PARAMS = {"early_stopping_rounds": 50, "verbose_eval": 100}

# ==========================================
# 5. Graph Optimization Parameters
# ==========================================
GRAPH_PARAMS = {
    # Base weight for the ML Anchor (Absolute Position)
    # This will be dynamically scaled by (1 / BiasUncertainty)
    "anchor_weight_base": 1.0,
    # Weight for the TDCP/Doppler Kinematic Edge (Relative Motion)
    # High confidence in Carrier Phase
    "kinematic_weight_tdcp": 100.0,
    # Lower confidence in Doppler fallback
    "kinematic_weight_doppler": 10.0,
    # RANSAC parameters for kinematic estimation
    "ransac_threshold": 0.5,  # m/s
    "min_inliers": 4,
}

# ==========================================
# 6. Random Seed
# ==========================================
SEED = 42
