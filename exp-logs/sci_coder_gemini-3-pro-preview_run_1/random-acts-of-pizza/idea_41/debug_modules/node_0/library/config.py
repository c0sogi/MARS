import os

# -----------------------------------------------------------------------------
# Global Configuration
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
N_JOBS = 12  # Utilizing available vCPUs
DEVICE = "cuda"  # GPU availability assumed based on environment description

# -----------------------------------------------------------------------------
# File Paths
# -----------------------------------------------------------------------------
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_41"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data paths (Metadata CSVs)
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache paths for processed features
CACHE_DIR = WORKING_DIR
CACHE_PATHS = {
    # Random Forest Features
    "rf_train": os.path.join(CACHE_DIR, "rf_features_train.npz"),
    "rf_val": os.path.join(CACHE_DIR, "rf_features_val.npz"),
    "rf_test": os.path.join(CACHE_DIR, "rf_features_test.npz"),
    # MLP Features
    "mlp_train": os.path.join(CACHE_DIR, "mlp_features_train.npz"),
    "mlp_val": os.path.join(CACHE_DIR, "mlp_features_val.npz"),
    "mlp_test": os.path.join(CACHE_DIR, "mlp_features_test.npz"),
    # Artifacts
    "top_k_subreddits": os.path.join(CACHE_DIR, "top_k_subreddits.json"),
    "tfidf_model": os.path.join(CACHE_DIR, "tfidf_model.pkl"),
    "scaler_model": os.path.join(CACHE_DIR, "scaler_model.pkl"),
}

# -----------------------------------------------------------------------------
# Feature Engineering Configuration
# -----------------------------------------------------------------------------
# Text Processing
TFIDF_MAX_FEATURES = 5000
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_TEXT_LENGTH = 512

# Community Profiling
TOP_K_COMMUNITIES = 50

# Numerical Features (Intersection of Train/Test + Useful Metadata)
NUMERIC_FEATURES = [
    "requester_account_age_in_days_at_request",
    "requester_days_since_first_post_on_raop_at_request",
    "requester_number_of_comments_at_request",
    "requester_number_of_comments_in_raop_at_request",
    "requester_number_of_posts_at_request",
    "requester_number_of_posts_on_raop_at_request",
    "requester_number_of_subreddits_at_request",
    "requester_upvotes_minus_downvotes_at_request",
    "requester_upvotes_plus_downvotes_at_request",
    "unix_timestamp_of_request",
]

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------

# Stream A: Global-Consistency Augmented Random Forest
RF_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
    "verbose": 0,
}

# Stream B: Community-Enhanced Skip-Gated MLP
MLP_PARAMS = {
    "input_embedding_dim": 384,  # Dimension for all-MiniLM-L6-v2
    "hidden_dim": 256,
    "dropout_prob": 0.5,  # Dropout for embedding layers
    "dropout_dense": 0.2,  # Dropout for dense layers
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 50,
    "patience": 15,  # Early stopping patience
    "scheduler_factor": 0.5,
    "scheduler_patience": 5,
}

# -----------------------------------------------------------------------------
# Ensemble Configuration
# -----------------------------------------------------------------------------
ENSEMBLE_WEIGHTS = {"rf": 0.5, "mlp": 0.5}
