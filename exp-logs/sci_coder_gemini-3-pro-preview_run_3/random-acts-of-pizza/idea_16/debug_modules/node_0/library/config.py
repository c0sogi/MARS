import os

# =============================================================================
# GLOBAL PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_16"
SUBMISSION_DIR = "./submission"

# Create necessary directories
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data File Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sampleSubmission.csv")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
SEED = 42
N_FOLDS = 5
TARGET_COL = "requester_received_pizza"
ID_COL = "request_id"

# Debugging / Development
# Set to None to use full dataset, or an integer (e.g., 100) to subsample for quick testing
DEBUG_SAMPLE_SIZE = None

# =============================================================================
# FEATURE EXTRACTION SETTINGS
# =============================================================================

# 1. Global Metadata Vector (Dense)
# Features to allow-list for the dense vector shared across all models
DENSE_FEATURE_COLS = [
    "requester_account_age_in_days_at_request",
    "requester_days_since_first_post_on_raop_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_request",
    # Derived text stats (calculated during processing)
    "request_text_len_char",
    "request_text_len_word",
    "request_title_len_char",
    "request_title_len_word",
]

# 2. Lexical View (TF-IDF on Request Text)
LEXICAL_VOCAB_SIZE = 3000
LEXICAL_NGRAM_RANGE = (1, 2)
LEXICAL_MIN_DF = 5
LEXICAL_SUBLINEAR_TF = True  # Scaling to reduce noise

# 3. Behavioral View (TF-IDF on Subreddits)
BEHAVIORAL_VOCAB_SIZE = 1000
BEHAVIORAL_NGRAM_RANGE = (1, 1)

# 4. Semantic View (SBERT Embeddings)
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
SBERT_BATCH_SIZE = 32
SBERT_DIM = 384

# =============================================================================
# MODEL HYPERPARAMETERS (LEVEL 1)
# =============================================================================

# 1. Lexical Bagger (Random Forest)
# Sparse Topology + Bagging + Regularization
RF_LEXICAL_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 2,  # Regularization to prevent memorizing sparse noise
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# 2. Behavioral Bagger (Random Forest)
# Sparse Topology + Bagging
RF_BEHAVIORAL_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# 3. Semantic Booster (XGBoost)
# Dense Topology + Boosting
XGB_SEMANTIC_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": SEED,
    "device": "cuda",  # Utilize A100 GPU
    "tree_method": "hist",  # Efficient histogram-based training
    "scale_pos_weight": 3.0,  # Handle class imbalance (~1:3)
    "verbosity": 0,
}
XGB_EARLY_STOPPING_ROUNDS = 100

# 4. Semantic Bagger (Random Forest)
# Dense Topology + Bagging (Restored for Diversity)
RF_SEMANTIC_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# 5. Contextual Anchor (Logistic Regression)
# Linear Topology + High Bias Regularizer
LOGREG_ANCHOR_PARAMS = {
    "penalty": "l2",
    "C": 0.1,  # Stronger regularization for the anchor
    "solver": "lbfgs",
    "class_weight": "balanced",
    "max_iter": 2000,
    "random_state": SEED,
    "n_jobs": -1,
}

# =============================================================================
# META-LEARNER HYPERPARAMETERS (LEVEL 2)
# =============================================================================

META_LEARNER_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "random_state": SEED,
    "n_jobs": -1,
}
