import os

# =============================================================================
# RUNTIME SETTINGS
# =============================================================================
RANDOM_SEED = 42
DEBUG = False  # Set to True to use a small subset of data for debugging
DATA_SAMPLE_SIZE = 100 if DEBUG else None  # Number of samples if DEBUG is True

# =============================================================================
# FILE PATHS
# =============================================================================
# Input Data Paths (Metadata CSVs)
TRAIN_PATH = "./metadata/train.csv"
VAL_PATH = "./metadata/val.csv"
TEST_PATH = "./metadata/test.csv"

# Cache Directory for Deterministic Processing
# Stores processed embeddings, features, and intermediate files
CACHE_DIR = "./working/idea_29/"
os.makedirs(CACHE_DIR, exist_ok=True)

# Submission Output Directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================
# SBERT Model for Semantic Embeddings (Stream B)
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Sequence Limits
MAX_SEQ_LEN_TEXT = 256  # Max tokens for request text
MAX_HISTORY_LEN = 50  # Max number of historical items (subreddits/posts) to process

# TF-IDF Configuration (Stream A)
# High-fidelity settings with continuous frequency weights
TFIDF_PARAMS = {
    "max_features": 5000,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "min_df": 5,
    "max_df": 0.9,
}

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# Stream A: Random Forest Params
# Scope-Restricted: Relies on Metadata + Sentiment + TF-IDF (No History)
# Low Regularization (min_samples_leaf=1) to capture fine-grained patterns
RF_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": 0,
}

# Stream B: Neural Network Params
# Alignment-Injected Dual-Query MLP
# High patience and sufficient epochs to allow attention mechanisms to stabilize
NN_PARAMS = {
    "hidden_dim": 256,
    "dropout_rate": 0.3,
    "learning_rate": 1e-4,
    "weight_decay": 1e-2,  # AdamW standard
    "batch_size": 32,
    "epochs": 50,
    "patience": 15,
    "device": "cuda",  # Preference for GPU
}

# Ensemble Averaging Weights
ENSEMBLE_WEIGHTS = {"rf": 0.5, "nn": 0.5}
