import os

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = "./working/idea_21"

# Ensure cache directory exists for deterministic processing artifacts
os.makedirs(CACHE_DIR, exist_ok=True)

# Data Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
SUBMISSION_PATH = "./submission/submission.csv"
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# ==========================================
# Global Constants & Reproducibility
# ==========================================
SEED = 42

# Coordinate Clamping (NYC Bounding Box)
# Format: (min_longitude, min_latitude, max_longitude, max_latitude)
# Limits chosen to cover NYC and major airports (JFK, LGA, EWR) while preventing
# linear extrapolation failures from corrupt GPS data.
NYC_BBOX = (-74.50, 40.50, -72.80, 41.80)

# Geohash Levels for Hierarchical Fingerprinting
# L5: Regional smoothing
# L6: Neighborhood context
# L7: Fine-grained memorization
GEOHASH_LEVELS = [5, 6, 7]

# Earth Radius for Distance Calculations (Haversine)
EARTH_RADIUS_KM = 6371.0

# ==========================================
# Dual-Hygiene Strategy Configuration
# ==========================================
# Wisdom Set (Background): Used exclusively for generating statistical fingerprints.
# Strict filtering ensures priors are not contaminated by noise or impossible trips.
STRICT_FILTER = {"min_fare": 2.50, "max_fare": 200.00, "max_fare_per_km": 10.00}

# Learner Set (Foreground): Used for training the model.
# Loose filtering retains valid high-fare outliers (Heavy Tail) to optimize RMSE.
LOOSE_FILTER = {"min_fare": 2.50}

# Learner Set Subsample Size
# A stable subsample of 5M rows is sufficient for the Learner Set given the
# robust priors provided by the full Wisdom Set.
LEARNER_SUBSET_SIZE = 5_000_000

# ==========================================
# Model Hyperparameters (XGBoost)
# ==========================================
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 9,  # Deep trees to capture complex spatial boundaries
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "n_estimators": 5000,  # High ceiling, controlled by early stopping
    "tree_method": "hist",  # Efficient histogram-based algorithm
    "device": "cuda",  # Leverage NVIDIA A100 GPU
    "n_jobs": 12,  # Utilize available vCPUs
    "random_state": SEED,
    "reg_alpha": 0.1,  # L1 regularization
    "reg_lambda": 1.0,  # L2 regularization
}

# Training Control
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 100
