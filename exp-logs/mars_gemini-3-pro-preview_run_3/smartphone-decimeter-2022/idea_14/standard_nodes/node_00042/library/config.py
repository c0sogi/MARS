import os

# --- Paths ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
CACHE_DIR = "./working/idea_14"
SUBMISSION_PATH = "./submission/submission.csv"

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Reproducibility ---
SEED = 42

# --- Cross-Validation ---
N_FOLDS = 5

# --- Feature Definitions ---
# These features correspond to the Physics-Based Feature Extraction described in the idea.
# 1. Projected Residual Forces (Net Error Force vector)
# 2. Geometry Conditioning (Covariance matrix elements / DOP proxies)
# 3. Global Signal Quality (Mean Cn0, Satellite Count)
# 4. IMU Dynamics (Acceleration stats)
FEATURES = [
    # Geometric Projections (The core innovation)
    "NetForce_E",
    "NetForce_N",
    "Cov_E",  # Sum of u_E^2
    "Cov_N",  # Sum of u_N^2
    "Cov_EN",  # Sum of u_E * u_N
    # Signal Quality
    "Cn0DbHz_mean",
    "Svid_count",
    # IMU Dynamics
    "Accel_mean",
    "Accel_std",
    # WLS Baseline Context (Optional but helpful for scaling)
    "Wls_Lat",
    "Wls_Lon",
]

# --- Model Hyperparameters ---
# LightGBM parameters optimized for regression with L1 loss (MAE)
LGBM_PARAMS = {
    "objective": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": -1,
}

# Training settings
NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 100
