import os
import torch

# ==========================================
# Paths and Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_47"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data File Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sampleSubmission.csv")
OUTPUT_SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Global Settings
# ==========================================
RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # For data loading

# ==========================================
# Feature Engineering Configuration
# ==========================================
# Number of top frequent subreddits to create binary indicators for
TOP_K_SUBREDDITS = 50

# Sentence-BERT model for embedding text
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# Columns containing text data
TEXT_COLS = ["request_text", "request_title"]

# Target column name
TARGET_COL = "requester_received_pizza"

# ==========================================
# Stream A: Random Forest Hyperparameters
# ==========================================
RF_N_ESTIMATORS = 500
RF_MIN_SAMPLES_LEAF = 1
RF_CLASS_WEIGHT = "balanced"
RF_N_JOBS = -1  # Use all available cores

# ==========================================
# Stream B: MLP Hyperparameters
# ==========================================
MLP_HIDDEN_DIM = 256
MLP_DROPOUT = 0.5  # Dropout for embeddings/main path
MLP_DROPOUT_DENSE = 0.2  # Dropout for dense layers
MLP_LEARNING_RATE = 1e-4
MLP_WEIGHT_DECAY = 1e-5
MLP_BATCH_SIZE = 32
MLP_NUM_EPOCHS = 50
MLP_PATIENCE = 15  # Early stopping patience

# ==========================================
# Ensemble Configuration
# ==========================================
ENSEMBLE_WEIGHT_RF = 0.5
ENSEMBLE_WEIGHT_MLP = 0.5
