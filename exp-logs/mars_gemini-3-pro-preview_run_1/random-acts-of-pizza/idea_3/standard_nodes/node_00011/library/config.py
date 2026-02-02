import os

# ==========================================
# Global Configuration & Paths
# ==========================================

# Input Data Paths (Generated Metadata)
TRAIN_PATH = "./metadata/train.csv"
VAL_PATH = "./metadata/val.csv"
TEST_PATH = "./metadata/test.csv"

# Output Paths
# Directory for caching intermediate processed data (parquet/npy)
CACHE_DIR = "./working/idea_3"
# Directory and file path for final submission
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Reproducibility
# ==========================================
RANDOM_SEED = 42

# ==========================================
# Model Hyperparameters
# ==========================================

# 1. Sparse-Tabular Learner (Random Forest)
# Uses Bag-of-Words and raw numerical features
RF_ESTIMATORS = 500
RF_CLASS_WEIGHT = "balanced"
RF_N_JOBS = -1

# 2. Dense-Semantic Learner (Logistic Regression)
# Uses Sentence Embeddings and scaled numerical features
LR_C = 10.0
LR_PENALTY = "l2"
LR_SOLVER = "lbfgs"
LR_MAX_ITER = 1000
LR_CLASS_WEIGHT = "balanced"

# 3. Fusion Strategy
# Weighted average of probabilities from RF and LR
FUSION_WEIGHT_RF = 0.4
FUSION_WEIGHT_LR = 0.6

# ==========================================
# Feature Engineering Configuration
# ==========================================

# Text Processing
# Using the edit-aware text column to prevent data leakage from "EDIT: Thanks" messages
TEXT_COLUMN = "request_text_edit_aware"

# Bag-of-Words Configuration (for Random Forest)
BOW_NGRAM_RANGE = (1, 2)  # Unigrams and Bigrams
BOW_MAX_FEATURES = 10000  # Limit vocabulary size to prevent memory issues

# Sentence Transformer Configuration (for Logistic Regression)
# Model optimized for semantic similarity
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# Numerical Features
# List of potential leakage columns to strictly exclude from features
LEAKAGE_COLUMNS = [
    "requester_received_pizza",
    "request_id",
    "source_file",
    "giver_username_if_known",
    "request_text",
    "request_title",
    "request_text_edit_aware",
    "unix_timestamp_of_request",  # Often redundant or leakage if not handled carefully
    "unix_timestamp_of_request_utc",
]
