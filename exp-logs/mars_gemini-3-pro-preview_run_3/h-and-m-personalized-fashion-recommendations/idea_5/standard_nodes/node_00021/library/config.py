import os
from pathlib import Path

# =============================================================================
# GLOBAL PATHS
# =============================================================================

# Base directories
INPUT_DIR = Path("./input")
METADATA_DIR = Path("./metadata")
WORKING_DIR = Path("./working/idea_5")
SUBMISSION_DIR = Path("./submission")

# Ensure writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Raw Data Files
ARTICLES_CSV = INPUT_DIR / "articles.csv"
CUSTOMERS_CSV = INPUT_DIR / "customers.csv"
TRANSACTIONS_CSV = INPUT_DIR / "transactions_train.csv"
SAMPLE_SUBMISSION_CSV = INPUT_DIR / "sample_submission.csv"

# Metadata Files (Pre-split and processed)
TRAIN_METADATA_PATH = METADATA_DIR / "train_metadata.parquet"
VAL_METADATA_PATH = METADATA_DIR / "val_metadata.parquet"
TEST_METADATA_PATH = METADATA_DIR / "test_metadata.parquet"

# =============================================================================
# CACHE FILE PATHS (Artifacts)
# =============================================================================

# Image Embeddings and Visual Graph
IMAGE_EMBEDDINGS_PATH = WORKING_DIR / "image_embeddings.npy"
ARTICLE_ID_MAP_PATH = WORKING_DIR / "article_id_map.npy"  # Maps article_id to index
VISUAL_GRAPH_PATH = WORKING_DIR / "visual_graph.npz"  # Sparse matrix

# Sequential Graph Artifacts
TRANSITION_MATRIX_PATH = WORKING_DIR / "transition_matrix.npz"
USER_HISTORY_PATH = (
    WORKING_DIR / "user_history.npz"
)  # Sparse user-item interaction matrix

# Ranker Datasets
RANKER_TRAIN_PATH = WORKING_DIR / "ranker_train.parquet"
RANKER_VAL_PATH = WORKING_DIR / "ranker_val.parquet"

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

# General
SEED = 42
NUM_CPU_WORKERS = 12  # Based on available vCPUs

# Stage 1: Retrieval (Vectorized Sparse Propagation)
# --------------------------------------------------
RETRIEVAL_TOP_K = 100  # Number of candidates to retrieve per user
VISUAL_KNN_K = 20  # Number of neighbors in the visual graph
RECENCY_WEEKS = 12  # Number of weeks of history to consider for transition matrix
TIME_DECAY_DAYS = 3  # Half-life for time decay in history aggregation

# Score Aggregation Weights
# Score = U.dot(T_seq) + LAMBDA_VISUAL * U.dot(T_vis) + ALPHA_HISTORY * U_history
LAMBDA_VISUAL = 0.15  # Weight for visual similarity signal
ALPHA_HISTORY = 1.5  # Weight for repurchase signal (history)

# Stage 2: Ranking (LightGBM)
# --------------------------------------------------
# Sliding window for ranker training data generation
RANKER_HISTORY_WEEKS = 10  # Weeks of history used to generate features for the ranker
RANKER_TARGET_WEEK_OFFSET = (
    0  # 0 means the very last week of training data is the target
)

# LightGBM Hyperparameters
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
    "verbose": -1,
    "seed": SEED,
    "n_jobs": NUM_CPU_WORKERS,
}

LGBM_FIT_PARAMS = {
    "callbacks": [],  # Will be populated with early_stopping in the trainer
}
EARLY_STOPPING_ROUNDS = 50

# Submission
# --------------------------------------------------
SUBMISSION_TOP_K = 12  # Final number of recommendations per user
OUTPUT_FILE = SUBMISSION_DIR / "submission.csv"
