import os
import torch
import numpy as np
import random

# ==========================================
# Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Working directory for intermediate artifacts (features, models, cache)
WORKING_DIR = "./working/idea_4"
# Directory for final submission
SUBMISSION_DIR = "./submission"

# Ensure writeable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Global Configuration
# ==========================================
SEED = 42
N_FOLDS = 5
NUM_WORKERS = 4  # Optimized for the 12 vCPU environment

# ==========================================
# Classical Model Configuration (Branch A)
# ==========================================
TFIDF_MIN_DF = 2
SVD_COMPONENTS = 100
WORD_NGRAM_RANGE = (1, 3)
CHAR_NGRAM_RANGE = (2, 5)

# ==========================================
# Neural Model Configuration (Branches B & C)
# ==========================================
# Branch B: Disentangled Attention
DEBERTA_MODEL = "microsoft/deberta-v3-large"
# Branch C: Absolute Attention
ROBERTA_MODEL = "roberta-large"

# Training Hyperparameters
MAX_LENGTH = 85  # Optimized for mean sentence length ~27
BATCH_SIZE = 16  # Ensures stable BN statistics
PATIENCE = 1  # Aggressive early stopping
EPOCHS = 3  # Sufficient for convergence with patience=1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 1

# ==========================================
# Hardware
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# Utility Functions
# ==========================================
def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
