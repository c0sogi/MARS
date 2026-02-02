import os
import torch
import random
import numpy as np

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
MODEL_NAME = "distilroberta-base"
MAX_LEN = 128  # Total sequence length
MAX_CODE_TOKENS = 32  # Number of tokens reserved for code context
MAX_MD_TOKENS = 96  # Number of tokens reserved for markdown content

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
# Adjust batch size based on available GPU memory (A100 40GB allows larger batches)
BATCH_SIZE = 64
VAL_BATCH_SIZE = 128
EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_WORKERS = 4
ACCUMULATE_GRAD_BATCHES = 1

# Early Stopping
PATIENCE = 3
MIN_DELTA = 1e-4

# -----------------------------------------------------------------------------
# Data Processing
# -----------------------------------------------------------------------------
VOCAB_SIZE_CODE = 10000  # Max features for TF-IDF on code cells
DEBUG_SAMPLE_SIZE = (
    None  # Set to an integer (e.g., 1000) for debugging, None for full run
)

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Hardware
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
