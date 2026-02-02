import os
import torch

# ------------------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------------------
SEED = 42

# ------------------------------------------------------------------------------
# Paths & Directories
# ------------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths (contain labels and splits, but no text)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Raw Data Paths (contain text)
TRAIN_TEXT_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_TEXT_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model.pth")
TOKENIZER_SAVE_PATH = os.path.join(CACHE_DIR, "tokenizer.json")
PROCESSED_DATA_PATH = os.path.join(CACHE_DIR, "processed_data.parquet")

# ------------------------------------------------------------------------------
# Data Configuration
# ------------------------------------------------------------------------------
ID_COL = "id"
TEXT_COL = "comment_text"
TARGET_COL = "target"

# Identity subgroups for bias mitigation (Auxiliary Targets)
IDENTITY_COLUMNS = [
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness",
]

# Text Preprocessing Hyperparameters
VOCAB_SIZE = 50000  # Size of the vocabulary (top frequent words)
MAX_LEN = 250  # Fixed sequence length (padding/truncation)
LOWERCASE = True  # Whether to lowercase text

# ------------------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------------------
EMBED_DIM = 300  # Dimension of the token embeddings
HIDDEN_DIM = 256  # Hidden dimension of the LSTM unit
LSTM_LAYERS = 1  # Number of stacked LSTM layers
BIDIRECTIONAL = True  # Whether to use a bidirectional LSTM
DROPOUT = 0.3  # Dropout probability

# ------------------------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------------------------
BATCH_SIZE = 512  # Batch size for training and inference
LEARNING_RATE = 1e-3  # Initial learning rate for Adam optimizer
EPOCHS = 5  # Maximum number of training epochs
PATIENCE = 1  # Early stopping patience (epochs without improvement)
AUX_LOSS_WEIGHT = 0.5  # Weight (lambda) for the auxiliary identity loss

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Number of subprocesses for data loading
