import os
from pathlib import Path

# ==========================================
# DIRECTORY CONFIGURATION
# ==========================================
INPUT_DIR = Path("./input")
METADATA_DIR = Path("./metadata")
WORKING_DIR = Path("./working/idea_2")
SUBMISSION_DIR = Path("./submission")

# Ensure working directories exist
WORKING_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# FILE PATHS
# ==========================================
# Raw Data
TRANSACTIONS_PATH = INPUT_DIR / "transactions_train.csv"
ARTICLES_PATH = INPUT_DIR / "articles.csv"
CUSTOMERS_PATH = INPUT_DIR / "customers.csv"
SAMPLE_SUBMISSION_PATH = INPUT_DIR / "sample_submission.csv"
IMAGES_DIR = INPUT_DIR / "images"

# Metadata (Split Definitions)
TRAIN_META_PATH = METADATA_DIR / "train.parquet"
VAL_META_PATH = METADATA_DIR / "val.parquet"
TEST_META_PATH = METADATA_DIR / "test.parquet"

# Caching / Intermediate Files
# Stage 1: Retrieval Artifacts
CACHE_COOCCURRENCE = WORKING_DIR / "cooccurrence_matrix.npz"
CACHE_GLOBAL_POPULARITY = WORKING_DIR / "global_popularity.npy"
CACHE_USER_MAP = WORKING_DIR / "user_map.parquet"
CACHE_ITEM_MAP = WORKING_DIR / "item_map.parquet"

# Stage 2: Feature Engineering Artifacts
CACHE_IMAGE_EMBEDDINGS = WORKING_DIR / "image_embeddings.npy"
CACHE_IMAGE_ID_MAP = WORKING_DIR / "image_id_map.npy"
CACHE_CANDIDATES_TRAIN = WORKING_DIR / "candidates_train.parquet"
CACHE_CANDIDATES_VAL = WORKING_DIR / "candidates_val.parquet"
CACHE_CANDIDATES_TEST = WORKING_DIR / "candidates_test.parquet"
CACHE_RANKER_DATASET = WORKING_DIR / "ranker_dataset.parquet"

# Model Artifacts
MODEL_PATH = WORKING_DIR / "lgbm_ranker.txt"

# Submission Output
SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"

# ==========================================
# DATA CONFIGURATION
# ==========================================
# Column Names
USER_ID_COL = "customer_id"
ITEM_ID_COL = "article_id"
DATE_COL = "t_dat"
IMAGE_PATH_COL = "image_path"
TARGET_COL = "purchased"

# ==========================================
# HYPERPARAMETERS
# ==========================================
SEED = 42
N_CPUS = 12

# Retrieval (Stage 1)
# Use 4 weeks of history for co-occurrence matrix calculation
RETRIEVAL_HISTORY_WEEKS = 4
# Validation period length
VAL_DAYS = 7
# Number of candidates to retrieve per user
TOP_K_RETRIEVAL = 100

# Ranking (Stage 2)
# LightGBM Hyperparameters
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 2000,
    "early_stopping_rounds": 50,
    "verbose": -1,
    "random_state": SEED,
    "n_jobs": N_CPUS,
}

# ==========================================
# FEATURES
# ==========================================
# Features from customers.csv
USER_FEATURES = ["age", "club_member_status", "fashion_news_frequency"]

# Features from articles.csv
ITEM_FEATURES = [
    "product_type_no",
    "graphical_appearance_no",
    "colour_group_code",
    "perceived_colour_value_id",
    "perceived_colour_master_id",
    "department_no",
    "index_group_no",
    "section_no",
    "garment_group_no",
]

# Computed Features
CONTEXT_FEATURES = ["cooccurrence_score", "visual_similarity"]

# All features for the model
RANKER_FEATURES = USER_FEATURES + ITEM_FEATURES + CONTEXT_FEATURES
