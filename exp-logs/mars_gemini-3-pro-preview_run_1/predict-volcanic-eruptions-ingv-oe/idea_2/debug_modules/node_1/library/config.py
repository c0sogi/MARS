import os
import hashlib
import json

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission Output Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# 2. GLOBAL SETTINGS
# ==========================================
RANDOM_SEED = 42
N_JOBS = 12  # Utilizing available 12 vCPUs

# ==========================================
# 3. FEATURE ENGINEERING CONFIGURATION
# ==========================================
# This dictionary defines the logic for the Hierarchical Feature Extraction.
# Any change here changes the config hash, forcing a cache refresh.
FEATURE_CONFIG = {
    # Sensor Definitions
    "sensors": [f"sensor_{i}" for i in range(1, 11)],
    # Data Preprocessing
    "sampling_rate": 100,  # Approx 100Hz (60k samples / 10 mins)
    "fill_na_strategy": "mean",  # Fill NaNs with segment mean
    # Level 1: Windowing Strategy (Local Dynamics)
    # Splitting 60,000 samples into windows
    "window_size": 6000,  # 10 windows per segment
    "window_overlap": 0,
    # Level 1: Features to extract per window
    "time_stats": [
        "mean",
        "std",
        "min",
        "max",
        "skew",
        "kurtosis",
        "q01",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
        "q99",
    ],
    "freq_stats": [
        "spectral_centroid",
        "dominant_freq",
        "spectral_power_mean",
        "spectral_power_std",
        "spectral_entropy",
    ],
    # Level 2: Aggregation Strategy (Global Dynamics)
    # How to summarize the sequence of window features
    "aggregation_stats": ["mean", "std", "min", "max", "skew"],
    # Spatial Features
    "use_spatial_correlation": True,  # Pearson correlation between sensors
}

# ==========================================
# 4. MODEL CONFIGURATION (LightGBM)
# ==========================================
MODEL_CONFIG = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 0.5,
    "min_data_in_leaf": 50,
    "early_stopping_rounds": 150,
    "verbosity": -1,
    "n_jobs": N_JOBS,
    "random_state": RANDOM_SEED,
}


# ==========================================
# 5. UTILITY FUNCTIONS
# ==========================================
def get_config_hash():
    """
    Generates a deterministic MD5 hash of the FEATURE_CONFIG dictionary.
    This hash is used to version cached feature files. If the configuration
    changes (e.g., window size or stats list), the hash changes, ensuring
    stale cache files are not loaded.
    """
    # Sort keys to ensure consistent serialization
    config_str = json.dumps(FEATURE_CONFIG, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()


def get_cache_path(subset_name):
    """
    Constructs the file path for cached features based on the subset name
    (e.g., 'train', 'val', 'test') and the current configuration hash.

    Args:
        subset_name (str): One of 'train', 'val', or 'test'.

    Returns:
        str: Absolute path to the parquet file.
    """
    config_hash = get_config_hash()
    filename = f"{subset_name}_features_{config_hash}.parquet"
    return os.path.join(WORKING_DIR, filename)
