import os
import torch
from pathlib import Path

# =============================================================================
# ENVIRONMENT & REPRODUCIBILITY
# =============================================================================
SEED = 42
NUM_WORKERS = 12  # Matches available vCPUs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# DIRECTORY STRUCTURE
# =============================================================================
INPUT_DIR = Path("./input")
METADATA_DIR = Path("./metadata")
WORKING_DIR = Path("./working/idea_8")
SUBMISSION_DIR = Path("./submission")

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
# Raw Inputs
ARTICLES_CSV = INPUT_DIR / "articles.csv"
CUSTOMERS_CSV = INPUT_DIR / "customers.csv"
TRANSACTIONS_CSV = INPUT_DIR / "transactions_train.csv"
SAMPLE_SUBMISSION_CSV = INPUT_DIR / "sample_submission.csv"
IMAGES_DIR = INPUT_DIR / "images"

# Metadata (Pre-split Parquet files)
TRAIN_METADATA = METADATA_DIR / "train_metadata.parquet"
VAL_METADATA = METADATA_DIR / "val_metadata.parquet"
TEST_METADATA = METADATA_DIR / "test_metadata.parquet"

# Intermediate Artifacts (Caching)
# Mappings
ARTICLE_ID_MAP_PATH = WORKING_DIR / "article_id_map.npy"
CUSTOMER_ID_MAP_PATH = WORKING_DIR / "customer_id_map.npy"

# Embeddings & Graphs
IMAGE_EMBEDDINGS_PATH = WORKING_DIR / "image_embeddings.npy"
VISUAL_GRAPH_PATH = WORKING_DIR / "visual_graph.npz"
SEQUENTIAL_GRAPH_PATH = WORKING_DIR / "sequential_graph.npz"

# Ranker Data
RANKER_TRAIN_DATA = WORKING_DIR / "ranker_train.parquet"
RANKER_VAL_DATA = WORKING_DIR / "ranker_val.parquet"
RANKER_TEST_DATA = WORKING_DIR / "ranker_test.parquet"

# Model & Output
LGBM_MODEL_PATH = WORKING_DIR / "lgbm_ranker.txt"
SUBMISSION_CSV = SUBMISSION_DIR / "submission.csv"

# =============================================================================
# DATA PARAMETERS
# =============================================================================
# Image Preprocessing
IMAGE_SIZE = (224, 224)
IMAGE_BATCH_SIZE = 128

# Temporal Settings
DATA_END_DATE = "2020-09-22"
TEST_WINDOW_DAYS = 7

# =============================================================================
# RETRIEVAL HYPERPARAMETERS (STAGE 1)
# =============================================================================
# Graph Construction
TRANSITION_HISTORY_WEEKS = 12  # Lookback for sequential graph (Strict Recency)
VISUAL_KNN_K = 20  # Neighbors for visual graph

# Propagation / Scoring
TOP_K_CANDIDATES = 100  # Candidates per user
LAMBDA_VIS = 0.2  # Weight for visual signal
ALPHA_HIST = 1.5  # Weight for repurchase signal

# =============================================================================
# RANKING HYPERPARAMETERS (STAGE 2)
# =============================================================================
# Sliding Window Strategy
# We use the last N weeks as target weeks for training the ranker.
# This creates multiple (History -> Target) splits to prevent overfitting.
RANKER_WINDOW_COUNT = 3  # Number of sliding windows (Splits A, B, etc.)
HISTORY_WINDOW_SIZE = 10  # Weeks of history used to predict the target week

# LightGBM Configuration
LGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [12],
    "boosting_type": "gbdt",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "random_state": SEED,
    "verbose": -1,
    "n_jobs": NUM_WORKERS,
}

# Training Control
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 50
