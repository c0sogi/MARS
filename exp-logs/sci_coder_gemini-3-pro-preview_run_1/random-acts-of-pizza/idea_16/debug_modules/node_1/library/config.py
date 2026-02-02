import os

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_16"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data File Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
NUM_WORKERS = 4  # Number of workers for data loading

# =============================================================================
# DATA PROCESSING HYPERPARAMETERS
# =============================================================================
# Text Processing (TF-IDF)
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# Latent Semantic Analysis (LSA) for Subreddit History
# Projects sparse subreddit history into dense latent community vectors
LSA_N_COMPONENTS = 20

# Sentence-BERT for Semantic Embeddings
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
SBERT_EMBEDDING_DIM = 384

# Columns to exclude from numerical/categorical feature sets
# (IDs, raw text, leakage, or fields processed separately)
DROP_COLS = [
    "request_id",
    "requester_received_pizza",
    "request_text",
    "request_title",
    "request_text_edit_aware",
    "source_file",
    "giver_username_if_known",
    "requester_subreddits_at_request",  # Handled via LSA
    "requester_username",
    "requester_user_flair",  # Potential leakage/high missingness
    "post_was_edited",
]

# =============================================================================
# MODEL HYPERPARAMETERS: STREAM A (RANDOM FOREST)
# =============================================================================
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# =============================================================================
# MODEL HYPERPARAMETERS: STREAM B (RESIDUAL-ATTENTION MLP)
# =============================================================================
MLP_PARAMS = {
    "hidden_dim": 256,
    "dropout_prob": 0.3,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 32,
    "epochs": 50,
    "patience": 15,  # High patience to allow attention mechanism to stabilize
    "scheduler_factor": 0.5,
    "scheduler_patience": 5,
}

# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================
ENSEMBLE_WEIGHTS = {"rf": 0.5, "mlp": 0.5}

# =============================================================================
# DEBUGGING
# =============================================================================
# Set to an integer (e.g., 100) to run on a small subset of data for debugging.
# Set to None for full training.
DEBUG_SAMPLE_SIZE = None
