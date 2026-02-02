import os
import torch

# =============================================================================
# GLOBAL PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_52"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

# Raw Data Paths (for fallback or text extraction if needed)
TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sampleSubmission.csv")

# Output Paths
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GENERAL CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # For DataLoader

# Debugging / Development
DEBUG = False
DEBUG_SAMPLE_SIZE = 100  # Number of samples to use if DEBUG is True

# =============================================================================
# DATA PROCESSING & FEATURE ENGINEERING
# =============================================================================
# Text Processing (TF-IDF for Random Forest)
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.9

# Semantic Embeddings (SBERT for MLP)
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_TEXT_LENGTH = 512  # Token limit for SBERT

# User History & Community Features
TOP_K_SUBREDDITS = 50
MAX_HISTORY_LENGTH = 10  # Max number of past posts to consider for history sequence

# =============================================================================
# MODEL HYPERPARAMETERS: MLP (Orthogonal Skip-Gated)
# =============================================================================
MLP_HIDDEN_DIM = 256
MLP_DROPOUT_EMB = 0.5  # Dropout for embedding layers
MLP_DROPOUT_DENSE = 0.2  # Dropout for dense layers
MLP_LEARNING_RATE = 1e-4
MLP_WEIGHT_DECAY = 1e-5
MLP_BATCH_SIZE = 32
MLP_EPOCHS = 50
MLP_PATIENCE = 15  # Early stopping patience

# =============================================================================
# MODEL HYPERPARAMETERS: RANDOM FOREST (Interaction-Enhanced)
# =============================================================================
RF_N_ESTIMATORS = 500
RF_MIN_SAMPLES_LEAF = 1
RF_CLASS_WEIGHT = "balanced"
RF_N_JOBS = -1
RF_MAX_DEPTH = None  # Allow full growth for interaction capture

# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================
# Weights for the weighted average ensemble [RandomForest, MLP]
ENSEMBLE_WEIGHTS = [0.5, 0.5]
