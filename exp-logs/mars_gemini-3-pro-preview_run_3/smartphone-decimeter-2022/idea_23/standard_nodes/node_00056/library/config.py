import os

# ==========================================
# 1. Directory and File Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_23"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Cached Data Paths
CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "features_train.parquet")
CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "features_val.parquet")
CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "features_test.parquet")

# ==========================================
# 2. Physical Constants & Global Settings
# ==========================================
SEED = 42
LIGHT_SPEED = 299792458.0  # m/s
L1_FREQ = 1575.42e6  # Hz
L5_FREQ = 1176.45e6  # Hz

# ==========================================
# 3. Model Hyperparameters (LightGBM)
# ==========================================
# Using Mean Absolute Error (regression_l1) to be robust against GNSS outliers
LGBM_PARAMS = {
    "objective": "regression_l1",
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "num_leaves": 128,
    "max_depth": -1,
    "min_child_samples": 20,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# Training Loop Settings
N_FOLDS = 5
EARLY_STOPPING_ROUNDS = 150
VERBOSE_EVAL = 100

# ==========================================
# 4. Hybrid Graph Optimization Parameters
# ==========================================
# Cost Function:
# J(x) = Sum(Huber(x - ML_Anchor)) + Sum(w_kin * ||(x_t - x_{t-1}) - delta_kin||^2)

# Robustness parameter for the Anchor term (meters)
# Errors larger than this transition from quadratic to linear penalty
HUBER_DELTA = 5.0

# Weight for the Anchor term (relative to shape term)
ANCHOR_WEIGHT = 1.0

# Weights for the Kinematic Shape term (Confidence)
# TDCP is mm-level accurate, so it gets a very high weight to enforce rigidity
SHAPE_WEIGHT_TDCP = 100.0

# Doppler is cm/s to m/s accurate, gets lower weight to allow some drift correction by anchors
SHAPE_WEIGHT_DOPPLER = 5.0

# RANSAC Parameters for Kinematic Estimation
RANSAC_THRESHOLD_TDCP = 0.02  # meters
RANSAC_THRESHOLD_DOPPLER = 0.5  # m/s
