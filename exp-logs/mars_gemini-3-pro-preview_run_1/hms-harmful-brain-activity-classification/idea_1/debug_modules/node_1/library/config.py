import os
import torch
import numpy as np
import random

# =============================================================================
# PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Cache directory for Idea 1 (Lightweight Spectrogram CRNN)
CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

# Specific data paths
TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
TEST_SPECS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")
TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")

# Metadata paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_CSV = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA PARAMETERS
# =============================================================================
# The model uses pre-computed spectrograms.
# We crop a central window of 60 seconds (out of 10 mins/600s available).
TIME_WINDOW = 60  # seconds
# The original spectrograms have width ~401. We resize or crop to 400.
FREQ_BINS = 400
# 4 regions: LL, RL, LP, RP
SPEC_CHANNELS = 4
# Target columns
TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
NUM_CLASSES = len(TARGET_COLS)

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 15
EARLY_STOPPING_PATIENCE = 4

# CRNN Specifics
CNN_FILTERS = [32, 64, 128]
RNN_HIDDEN_SIZE = 128
RNN_LAYERS = 1
DROPOUT = 0.5

# =============================================================================
# HARDWARE & REPRODUCIBILITY
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Number of workers for DataLoader.
# Set to 0 or small number to avoid shared memory issues in some envs,
# but 2-4 is usually good for speed.
NUM_WORKERS = 2
SEED = 42


def set_seed(seed: int = SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_directories():
    """
    Ensures that working, cache, and submission directories exist.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)


# Initialize environment immediately upon import
set_seed(SEED)
setup_directories()
