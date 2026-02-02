import os

# =============================================================================
# 1. File Paths & Directories
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_10"

# Ensure the working directory exists for caching
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# 2. Reproducibility
# =============================================================================
SEED = 42

# =============================================================================
# 3. Model Hyperparameters (LightGBM)
# =============================================================================
# Using MAE (Mean Absolute Error) to be robust against GPS outliers
LGBM_PARAMS = {
    "objective": "mae",
    "n_estimators": 5000,  # High number, controlled by early stopping
    "learning_rate": 0.05,
    "num_leaves": 128,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# =============================================================================
# 4. Global Trajectory Optimization Hyperparameters
# =============================================================================
# Physics Loss Weight (Lambda)
# Loss = ||x - x_ml||_1 + lambda * ||(x_t - x_{t-1}) - v_doppler * dt||_2^2
# Balances trust in ML prediction (L1) vs Kinematic consistency (L2)
PHYSICS_LAMBDA = 2.0

# Optimizer settings for the PyTorch trajectory refinement
OPTIMIZER_LR = 0.1
OPTIMIZER_EPOCHS = 500

# =============================================================================
# 5. Data Loading Configuration
# =============================================================================
# Relevant columns to load from device_gnss.csv
GNSS_COLS = [
    "utcTimeMillis",
    "TimeNanos",
    "FullBiasNanos",
    "BiasNanos",
    "Cn0DbHz",
    "PseudorangeRateMetersPerSecond",
    "PseudorangeRateUncertaintyMetersPerSecond",
    "SvElevationDegrees",
    "SvAzimuthDegrees",
    "SvVelocityXEcefMetersPerSecond",
    "SvVelocityYEcefMetersPerSecond",
    "SvVelocityZEcefMetersPerSecond",
    "SvPositionXEcefMeters",
    "SvPositionYEcefMeters",
    "SvPositionZEcefMeters",
    "RawPseudorangeMeters",
    "RawPseudorangeUncertaintyMeters",
    "WlsPositionXEcefMeters",
    "WlsPositionYEcefMeters",
    "WlsPositionZEcefMeters",
    "Svid",
    "ConstellationType",
]

# Relevant columns to load from device_imu.csv
IMU_COLS = [
    "utcTimeMillis",
    "MeasurementX",
    "MeasurementY",
    "MeasurementZ",
    "MessageType",
]
