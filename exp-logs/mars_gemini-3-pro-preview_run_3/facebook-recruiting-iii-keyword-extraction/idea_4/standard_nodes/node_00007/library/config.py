import os
import torch

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_4"
SUBMISSION_DIR = "./submission"

# Input Files (Metadata)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Cache & Artifacts
# These paths are used to save/load processed data and models
TOKENIZER_PATH = os.path.join(WORKING_DIR, "tokenizer.json")
LABEL_ENCODER_PATH = os.path.join(WORKING_DIR, "mlb.joblib")
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_data.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_data.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_data.parquet")
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "hybrid_cnn_transformer.pth")

# Submission
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42

# =============================================================================
# DATA HYPERPARAMETERS
# =============================================================================
VOCAB_SIZE = 50000  # Top N most frequent words
MAX_LEN = 300  # Fixed sequence length for Title + Body
TOP_K_TAGS = 3000  # Number of most frequent tags to predict

# Debugging / Development
# Set to an integer (e.g., 10000) to limit dataset size for fast debugging.
# Set to None for the full training run.
DEBUG_SAMPLE_SIZE = None

# =============================================================================
# MODEL ARCHITECTURE HYPERPARAMETERS
# =============================================================================
# Embedding
EMBED_DIM = 256

# CNN (Local Feature Extraction)
CNN_FILTERS = 256
CNN_KERNEL_SIZE = 3

# Transformer (Global Context)
TRANSFORMER_LAYERS = 2
NUM_HEADS = 4
TRANSFORMER_FF_DIM = 1024  # Feed-forward dimension
DROPOUT = 0.1

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 512  # Optimized for A100 40GB
LEARNING_RATE = 1e-3  # AdamW default
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 10
PATIENCE = 3  # Early stopping patience
WARMUP_RATIO = 0.1  # % of steps for linear warmup

# Threshold for converting probabilities to tags
PREDICTION_THRESHOLD = 0.3

# =============================================================================
# HARDWARE
# =============================================================================
NUM_WORKERS = 12  # Matches available vCPUs
PIN_MEMORY = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
