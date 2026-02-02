import os
import torch

# -----------------------------------------------------------------------------
# Global Directories and Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_15"
SUBMISSION_DIR = "./submission"

# Input Data Paths (Metadata)
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Submission Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Global Constants
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Semantic Anchors
# -----------------------------------------------------------------------------
# These anchor texts are used to create semantic profiles of user history.
# We calculate the cosine similarity between a user's subreddit history centroid
# and these anchor embeddings to generate dense behavioral features.
SEMANTIC_ANCHORS = {
    "Financial_Assistance": "loan borrow money broke paycheck rent bills debt help poor poverty need assistance",
    "Altruism_Giving": "gift pizza random acts pay it forward offer giving kindness charity donate",
    "Gaming_Hobbies": "game steam play xbox ps4 nintendo pc gaming fun entertainment console",
    "General_Discussion": "chat discuss question talk reddit community social interaction news politics",
}

# -----------------------------------------------------------------------------
# Feature Engineering Configuration
# -----------------------------------------------------------------------------
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_TEXT_LENGTH = 512

# TF-IDF Configuration for Random Forest
# We use separate TF-IDF vectorizers for Title and Body
TFIDF_MAX_FEATURES = 1000

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------

# Stream A: Semantic Anchor Random Forest
RF_PARAMS = {
    "n_estimators": 500,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": 0,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
}

# Stream B: Attention-Based Residual MLP
MLP_PARAMS = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "epochs": 50,
    "early_stopping_patience": 15,
    "dropout_rate": 0.3,
    "hidden_dim": 256,
    "embedding_dim": 384,  # Output dimension of all-MiniLM-L6-v2
    "random_state": RANDOM_STATE,
}

# Ensemble Configuration
ENSEMBLE_WEIGHTS = {"rf": 0.5, "mlp": 0.5}
