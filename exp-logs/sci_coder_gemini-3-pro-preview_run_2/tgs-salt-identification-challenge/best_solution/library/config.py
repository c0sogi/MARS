import os
import random
import numpy as np
import torch

# -----------------------------------------------------------------------------
# File Paths & Directories
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_8"
SUBMISSION_DIR = "./submission"

TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Parameters
# -----------------------------------------------------------------------------
ORIG_SIZE = 101
IMG_SIZE = 128  # Pad to power of 2 (128) for network architecture compatibility
CHANNELS = 1  # Input channels (Grayscale with modified first layer)

# Normalization (ImageNet Statistics - Averaged for Grayscale)
IMAGENET_MEAN = [0.449]
IMAGENET_STD = [0.226]

# -----------------------------------------------------------------------------
# Model Parameters
# -----------------------------------------------------------------------------
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
EPOCHS = 50
NUM_WORKERS = 4

# Debugging / Development Control
# Set these to an integer (e.g., 100) to limit dataset size for quick testing
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None

# -----------------------------------------------------------------------------
# Augmentation Parameters
# -----------------------------------------------------------------------------
AUG_PROB = 0.2
ELASTIC_ALPHA = 120
ELASTIC_SIGMA = 6
ELASTIC_ALPHA_AFFINE = 3.6  # 120 * 0.03


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Apply seed immediately upon import
seed_everything(SEED)
