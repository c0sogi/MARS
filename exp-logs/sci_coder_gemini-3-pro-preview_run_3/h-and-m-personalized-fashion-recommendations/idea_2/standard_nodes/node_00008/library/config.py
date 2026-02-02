import os
from pathlib import Path

# =============================================================================
# GLOBAL PATHS
# =============================================================================

# Base Directories
INPUT_DIR = Path("./input")
METADATA_DIR = Path("./metadata")
WORKING_DIR = Path("./working/idea_2")
SUBMISSION_DIR = Path("./submission")

# Ensure writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Raw Data Files
ARTICLES_CSV = INPUT_DIR / "articles.csv"
CUSTOMERS_CSV = INPUT_DIR / "customers.csv"
TRANSACTIONS_TRAIN_CSV = INPUT_DIR / "transactions_train.csv"
SAMPLE_SUBMISSION_CSV = INPUT_DIR / "sample_submission.csv"
IMAGES_DIR = INPUT_DIR / "images"

# Metadata Files (Pre-split Parquet files)
TRAIN_METADATA_PATH = METADATA_DIR / "train_metadata.parquet"
VAL_METADATA_PATH = METADATA_DIR / "val_metadata.parquet"
TEST_METADATA_PATH = METADATA_DIR / "test_metadata.parquet"

# Output Files
SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"

# =============================================================================
# DATA CONSTANTS
# =============================================================================

# Column Names
USER_COL = "customer_id"
ITEM_COL = "article_id"
DATE_COL = "t_dat"
PRICE_COL = "price"
SALES_CHANNEL_COL = "sales_channel_id"
IMAGE_PATH_COL = "image_path"
PREDICTION_COL = "prediction"

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# General
SEED = 42
N_CORES = 12  # Number of vCPUs available

# Stage 1: Retrieval (Sparse Transition Graph)
# Number of candidates to retrieve per user for the ranking stage
TOP_K_CANDIDATES = 100

# Time Decay factor for transition matrix construction
# Formula: weight = 1 / (1 + DECAY_RATE * days_diff)
DECAY_RATE = 0.01

# Weight for the user's own history (repurchase) vs global transitions
# Score = T_score + HISTORY_WEIGHT * History_score
HISTORY_WEIGHT = 1.5

# Stage 2: Ranking (LightGBM)
# Number of days to use for the validation set (simulating the test period)
VAL_DAYS = 7

# LightGBM Parameters
LGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "map",
    "eval_at": 12,
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
    "random_state": SEED,
    "n_jobs": N_CORES,
    "early_stopping_rounds": 50,
}

# =============================================================================
# CACHE PATHS
# =============================================================================
# Paths for intermediate artifacts to speed up iterative development
CACHE_TRANSITION_MATRIX = WORKING_DIR / "transition_matrix.npz"
CACHE_ARTICLE_MAP = WORKING_DIR / "article_map.npy"  # Map article_id to index
CACHE_CANDIDATES_TRAIN = WORKING_DIR / "candidates_train.parquet"
CACHE_CANDIDATES_TEST = WORKING_DIR / "candidates_test.parquet"
CACHE_FEATURES_TRAIN = WORKING_DIR / "features_train.parquet"
CACHE_FEATURES_TEST = WORKING_DIR / "features_test.parquet"
CACHE_RANKER_MODEL = WORKING_DIR / "lgbm_ranker.txt"
