import os

# --- General Configuration ---
SEED = 42
DEBUG = False  # Set to True to run on a small subset of data for debugging

# --- Directory Paths ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_22"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# --- File Paths ---
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_OUTPUT_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# --- GNSS Physics Constants ---
LIGHT_SPEED = 299792458.0  # m/s

# Carrier Frequencies (Hz)
FREQ_GPS_L1 = 1575.42e6
FREQ_GPS_L5 = 1176.45e6

# --- Split-Band Signal Definitions ---
# L1 Band: Legacy signals, wider availability but higher multipath susceptibility
L1_SIGNALS = {"GPS_L1", "GAL_E1", "GLO_G1", "QZS_J1", "BDS_B1I", "BDS_B1C"}

# L5 Band: Modern signals, higher chip rate, physically robust to multipath
L5_SIGNALS = {"GPS_L5", "GAL_E5A", "BDS_B2A", "QZS_J5"}

# --- Model Hyperparameters (LightGBM) ---
# The model predicts ENU (East-North-Up) residuals relative to the WLS baseline.
# We use Mean Absolute Error (MAE) to be robust against heavy outliers in urban canyons.
LGBM_PARAMS = {
    "objective": "mae",
    "boosting_type": "gbdt",
    "n_estimators": 6000,
    "learning_rate": 0.03,
    "num_leaves": 128,
    "max_depth": -1,
    "min_child_samples": 20,
    "colsample_bytree": 0.7,
    "subsample": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": -1,
}

# Training Control
EARLY_STOPPING_ROUNDS = 150
VERBOSE_EVAL = 100

# --- Graph Optimization Parameters ---
# The global trajectory is solved by minimizing:
# J = sum(Huber(x_t - P_ML)) + sum(lambda * S_t * ||odom_error||^2)

# Anchor Term: Huber Loss Delta (meters)
# Residuals < DELTA are squared (L2), > DELTA are linear (L1).
# This allows the trajectory to snap to ML predictions when they are consistent,
# but break away ("robustness") when the ML prediction is a massive outlier.
HUBER_DELTA = 3.0

# Odometry Term: Base Weight (Lambda)
# Controls the stiffness of the trajectory. Higher values enforce kinematic consistency
# more strongly over the absolute ML position estimates.
ODOM_WEIGHT = 8.0

# Reliability Scores (S_t)
# These modulate the ODOM_WEIGHT based on the quality of the odometry estimation.
# High confidence is assigned when Time-Differenced Carrier Phase (TDCP) RANSAC succeeds.
ODOM_RELIABILITY_HIGH = 1.0
# Low confidence is assigned when falling back to Doppler-based velocity or IMU.
ODOM_RELIABILITY_LOW = 0.05

# RANSAC Parameters for Odometry Estimation
RANSAC_THRESHOLD_METERS = 0.3  # Tight threshold for carrier phase differencing
RANSAC_MIN_SAMPLES = 5  # Minimum number of satellites to accept a TDCP fix
