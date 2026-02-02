import os
import torch

# -----------------------------------------------------------------------------
# Directory Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_4"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# File Paths
# -----------------------------------------------------------------------------
# Metadata files (contain labels and split info, but no text)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Source files (contain the actual text)
# Note: Metadata 'source_file' column points to these filenames
TRAIN_TEXT_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_TEXT_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Saved Model Path
SAVED_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
MODEL_NAME = "roberta-base"
MAX_LEN = 512  # Max sequence length for Transformer
DROPOUT = 0.2  # Spatial dropout rate

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
LR = 2e-5
NUM_EPOCHS = 2
NUM_WORKERS = 4

# Loss Weights
# High weight for identity task to force disentanglement
AUX_LOSS_WEIGHT = 0.8

# -----------------------------------------------------------------------------
# Data Definitions
# -----------------------------------------------------------------------------
TARGET_COL = "target"

# Identity columns used for the auxiliary task and bias evaluation
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

# Auxiliary toxicity subtypes (available in train, not predicted for submission)
AUX_TOXICITY_COLUMNS = [
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

# -----------------------------------------------------------------------------
# Hardware
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
