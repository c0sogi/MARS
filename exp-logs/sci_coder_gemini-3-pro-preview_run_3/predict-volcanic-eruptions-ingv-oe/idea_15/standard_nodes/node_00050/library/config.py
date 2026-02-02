import os

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_16"
SUBMISSION_DIR = "./submission"

# Ensure necessary writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Paths (Parquet files for intermediate feature storage)
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

# Model Persistence Path
MODEL_PATH = os.path.join(WORKING_DIR, "lgbm_model.txt")

# ==========================================
# Global Constants & Reproducibility
# ==========================================
SEED = 42
NUM_SENSORS = 10
N_FOLDS = 5

# Debugging / Development Flags
DEBUG = False  # Set to True to run on a small subset of data
DEBUG_SAMPLE_SIZE = 200  # Number of files to process if DEBUG is True

# ==========================================
# Signal Processing Hyperparameters
# ==========================================
# 1. Trend Extraction (Savitzky-Golay Filter)
# Window size must be odd and > 20 as per strategy
SG_WINDOW_SIZE = 51
SG_POLY_ORDER = 2

# 2. Texture Extraction (Wavelet Transform)
WAVELET_TYPE = "db4"

# 3. Spectral Analysis (Power Spectral Density)
# Assuming 100Hz sampling rate typical for this data type
SAMPLING_RATE = 100
PSD_BANDS = {
    "low": (0.1, 3.0),  # Low frequency / Long period movements
    "mid": (3.0, 10.0),  # Typical volcanic tremor range
    "high": (10.0, 45.0),  # High frequency noise/events
}

# 4. Window Statistics
# Size of sub-windows for calculating flattened temporal stats
WINDOW_SIZE = 5000

# 5. Feature Engineering
# Quantiles to compute for Trend and Raw distributions
QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

# ==========================================
# Model Hyperparameters (LightGBM)
# ==========================================
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "n_estimators": 10000,  # High number of estimators
    "learning_rate": 0.01,  # Lower learning rate for better generalization
    "num_leaves": 128,  # Increased complexity for high-dimensional feature set
    "max_depth": -1,  # No depth limit, controlled by num_leaves
    "feature_fraction": 0.8,  # Randomly select 80% of features per iteration
    "bagging_fraction": 0.8,  # Randomly select 80% of data per iteration
    "bagging_freq": 1,  # Perform bagging every iteration
    "lambda_l1": 0.5,  # L1 Regularization
    "lambda_l2": 0.5,  # L2 Regularization
    "n_jobs": 12,  # Utilize available vCPUs
    "seed": SEED,
    "verbose": -1,  # Silent mode
    "early_stopping_rounds": 200,
}
