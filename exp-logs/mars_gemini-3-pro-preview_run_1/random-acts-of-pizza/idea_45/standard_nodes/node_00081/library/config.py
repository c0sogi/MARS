import os

# =============================================================================
# GLOBAL PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_45"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Raw JSON Paths (for complex nested structures if needed)
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")
# Note: The metadata generation script handles the split logic.
# We rely on CSVs for the split indices/labels.

# Cache File Paths
CACHE_RF_FEATURES = os.path.join(WORKING_DIR, "rf_features.parquet")
CACHE_MLP_FEATURES = os.path.join(WORKING_DIR, "mlp_features.npz")
CACHE_SBERT_EMBEDDINGS = os.path.join(WORKING_DIR, "sbert_embeddings.npz")
CACHE_TOPK_FLAGS = os.path.join(WORKING_DIR, "topk_flags.parquet")

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
SEED = 42
NUM_WORKERS = 4  # For data loading

# =============================================================================
# FEATURE COLUMNS
# =============================================================================
ID_COL = "request_id"
TARGET_COL = "requester_received_pizza"

# Text Columns
TEXT_COLS = ["request_title", "request_text_edit_aware"]

# Raw Numerical Metadata Columns (Available in Train/Test)
# We exclude leakage columns and post-request data (retrieval time data)
# strictly adhering to 'at_request' suffix where possible.
RAW_NUMERIC_COLS = [
    "requester_account_age_in_days_at_request",
    "requester_days_since_first_post_on_raop_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_request",
]

# Columns to construct derived features from
LIST_COL = "requester_subreddits_at_request"

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# 1. Random Forest (Stream A)
RF_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# 2. MLP (Stream B)
MLP_PARAMS = {
    "hidden_dim": 256,
    "film_dim": 128,  # Dimension for FiLM modulation
    "dropout_emb": 0.5,  # Dropout for embeddings
    "dropout_dense": 0.2,  # Dropout for dense layers
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 50,
    "patience": 15,  # Early stopping patience
    "device": "cuda",  # Will fallback to cpu if not available
}

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

# Text Processing
TEXT_CONFIG = {
    "tfidf_max_features": 5000,
    "tfidf_ngram_range": (1, 2),
    "sbert_model_name": "all-MiniLM-L6-v2",  # Fast and effective
    "embedding_dim": 384,  # Dimension of SBERT embeddings
}

# Top-K Subreddits
TOP_K_CONFIG = {"k": 50}

# Interaction-Projected Features (for Random Forest)
# Definitions for explicit feature creation
INTERACTION_FEATURES = [
    {
        "name": "interaction_title_consistency_age",
        "formula": "title_consistency * log(1 + requester_account_age_in_days_at_request)",
    },
    {
        "name": "interaction_body_consistency_upvote_ratio",
        "formula": "body_consistency * (requester_upvotes_minus_downvotes_at_request / (requester_upvotes_plus_downvotes_at_request + 1))",
    },
    {
        "name": "interaction_title_consistency_topk_sum",
        "formula": "title_consistency * sum_top_k_flags",
    },
]
