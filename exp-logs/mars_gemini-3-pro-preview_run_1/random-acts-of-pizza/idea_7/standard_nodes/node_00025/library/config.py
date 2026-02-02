import os

# ==========================================
# Paths and Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_8")
SUBMISSION_DIR = "./submission"

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Dataset Paths (using generated metadata)
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Global Constants
# ==========================================
RANDOM_STATE = 42
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"  # Produces 384-dim embeddings

# ==========================================
# Domain Lexicons for Feature Engineering
# ==========================================
# Innovation: Explicitly feed semantic concepts as dense numerical signals
LEXICONS = {
    "reciprocity": ["pay", "return", "check", "friday"],
    "desperation": ["broke", "starving", "homeless", "job"],
    "gratitude": ["thanks", "appreciate"],
}

# ==========================================
# Model Hyperparameters
# ==========================================

# Learner A: Augmented Random Forest
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Learner B: Domain-Aware Dual-Branch MLP
MLP_PARAMS = {
    "sbert_dim": 384,
    "hidden_dim_text": 128,
    "hidden_dim_meta": 64,
    "fusion_dim": 64,
    "dropout_text": 0.5,  # High dropout for distributed representation
    "dropout_meta": 0.1,  # Low dropout for high-signal features
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
}

# ==========================================
# Training Configuration
# ==========================================
TRAIN_PARAMS = {
    "batch_size": 32,
    "epochs": 60,  # 50+ as required
    "patience": 15,  # High patience to survive warm-up phase
    "num_workers": 2,
}
