import os
import torch

# ==========================================
# Global Configuration
# ==========================================

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Path Configuration
# ==========================================

# Input Metadata Paths (Generated previously)
METADATA_DIR = "./metadata"
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

# Raw Data Paths
INPUT_DIR = "./input"
# Note: The actual file names might vary slightly based on the dataset structure provided in description,
# but these are the standard targets.
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

# Working Directory for Caching and Artifacts
# Specific to the current idea iteration to avoid conflicts
WORKING_DIR = "./working/idea_54"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission Directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Feature Engineering Configuration
# ==========================================

# Text Processing
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
TFIDF_VOCAB_SIZE = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# Community / History Profiling
TOP_K_SUBREDDITS = 50  # Number of top frequent subreddits to track as binary flags

# ==========================================
# Model Configuration: Random Forest (Stream A)
# ==========================================

RF_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ==========================================
# Model Configuration: MLP (Stream B)
# ==========================================

# Architecture Dimensions
MLP_HIDDEN_DIM = 128
MLP_DROPOUT_EMB = 0.5
MLP_DROPOUT_DENSE = 0.2

# Training Hyperparameters
MLP_BATCH_SIZE = 32
MLP_LEARNING_RATE = 1e-4
MLP_WEIGHT_DECAY = 1e-2
MLP_EPOCHS = 50
MLP_PATIENCE = 15  # Early stopping patience

# ==========================================
# Ensemble Configuration
# ==========================================

ENSEMBLE_WEIGHTS = {"rf": 0.5, "mlp": 0.5}
