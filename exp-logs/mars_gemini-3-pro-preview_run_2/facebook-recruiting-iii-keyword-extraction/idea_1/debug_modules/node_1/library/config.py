import os
import torch

# ==========================================
# Global Random Seed
# ==========================================
SEED = 42

# ==========================================
# File Paths and Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_1"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Paths for Preprocessing
# Using .json for mappings and .parquet for processed dataframes
VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.json")
TAG_MAP_PATH = os.path.join(WORKING_DIR, "tag_map.json")
TRAIN_PROCESSED_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
VAL_PROCESSED_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
TEST_PROCESSED_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

# Model Artifacts
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "dan_model.pth")

# ==========================================
# Data Processing Hyperparameters
# ==========================================
# Vocabulary
MAX_VOCAB_SIZE = 50000  # Size of the vocabulary (top N frequent words)
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"

# Tags
NUM_TARGET_TAGS = 5000  # Number of top frequent tags to predict
TAG_THRESHOLD = (
    5  # Minimum frequency for a tag to be included (if not using fixed top K)
)

# Sequence
MAX_SEQ_LEN = 300  # Maximum number of tokens per input sequence (Title + Body)

# Debugging
DEBUG = False  # Set to True to use a small subset of data
DEBUG_SAMPLE_SIZE = 10000  # Number of samples to use in debug mode

# ==========================================
# Model Architecture Hyperparameters
# ==========================================
EMBEDDING_DIM = 300  # Dimension of word embeddings
HIDDEN_DIMS = [1024, 512]  # Dimensions of hidden dense layers
DROPOUT_RATE = 0.2  # Dropout probability
OUTPUT_DIM = NUM_TARGET_TAGS

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 1024  # Large batch size for efficiency with DAN
LEARNING_RATE = 1e-3  # Adam optimizer learning rate
NUM_EPOCHS = 20  # Maximum number of training epochs
EARLY_STOPPING_PATIENCE = 3  # Stop if validation loss doesn't improve for N epochs
PREDICTION_THRESHOLD = (
    0.25  # Probability threshold for binary tag prediction (can be tuned)
)

# ==========================================
# Hardware Configuration
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # Number of subprocesses for data loading
PIN_MEMORY = True  # Pin memory for faster host-to-device transfer
