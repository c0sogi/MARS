import os

# -----------------------------------------------------------------------------
# Directories and Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Cache Directory for Idea 57 (Deterministic Data Processing)
CACHE_DIR = os.path.join(WORKING_DIR, "idea_57")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission Output Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Global Settings
# -----------------------------------------------------------------------------
RANDOM_STATE = 42

# -----------------------------------------------------------------------------
# Feature Engineering Configuration
# -----------------------------------------------------------------------------
# SBERT model for semantic embeddings (Title, Body, Subreddits)
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# Number of top subreddits to track as binary indicators (Top-K Community Signal)
TOP_K_SUBREDDITS = 50

# TF-IDF Configuration for Random Forest (High-Fidelity Text Features)
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# -----------------------------------------------------------------------------
# Model Hyperparameters: Random Forest (Stream A)
# -----------------------------------------------------------------------------
# Configured for Interaction-Projected Top-K Random Forest
RF_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 1,  # Low regularization to capture fine-grained signals
    "class_weight": "balanced",  # Handle class imbalance
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": 0,
}

# -----------------------------------------------------------------------------
# Model Hyperparameters: MLP (Stream B)
# -----------------------------------------------------------------------------
# Configured for Topology-Aware Non-Linear Skip-Gated MLP

# Training Loop
MLP_EPOCHS = 50
MLP_PATIENCE = 15  # High patience for complex convergence
MLP_BATCH_SIZE = 32

# Optimizer
MLP_LEARNING_RATE = 1e-4
MLP_WEIGHT_DECAY = 1e-2

# Architecture / Regularization
MLP_HIDDEN_DIM = 256  # Dimension for semantic projection and gating
MLP_DROPOUT_EMB = 0.5  # Higher dropout on embeddings
MLP_DROPOUT_DENSE = 0.2  # Lower dropout on dense layers

# -----------------------------------------------------------------------------
# Ensemble Configuration
# -----------------------------------------------------------------------------
# Simple Weighted Average
ENSEMBLE_WEIGHT_RF = 0.5
ENSEMBLE_WEIGHT_MLP = 0.5
