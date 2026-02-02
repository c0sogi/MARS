import os
import torch
import numpy as np
import random

# ====================================================
# Directory Settings
# ====================================================
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

# Metadata paths (pre-generated)
METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Working directories for Idea 8 (Recurrent U-Net + Aspect Ratio Preserving)
WORKING_DIR = "./working/idea_8"
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
PREDICTION_DIR = os.path.join(WORKING_DIR, "predictions")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ====================================================
# Data & Preprocessing Hyperparameters
# ====================================================
SEED = 42
IMG_SIZE = (256, 256)  # Resize target (Height, Width)
SEQ_LEN = 5  # Sequence length for Recurrent U-Net (z-2, z-1, z, z+1, z+2)

# Normalization (Robust Percentile Scaling)
LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0

# Class definitions
CLASSES = ["large_bowel", "small_bowel", "stomach"]
NUM_CLASSES = len(CLASSES)

# ====================================================
# Model Hyperparameters
# ====================================================
BACKBONE = "resnet34"
IN_CHANNELS = 3  # ResNet backbone expects 3 channels (we replicate the slice)
PRETRAINED = True

# ====================================================
# Training Hyperparameters
# ====================================================
BATCH_SIZE = 16  # Adjusted for Sequence input and A100 memory
NUM_WORKERS = 4
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
MAX_GRAD_NORM = 1000

# Loss weights (0.4 Dice, 0.6 Hausdorff approximation via BCE/Focal)
# Note: Hausdorff is hard to optimize directly, usually BCE+Dice is used as proxy
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5


# ====================================================
# Utility Functions
# ====================================================
def set_seed(seed=SEED):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set seed immediately upon import
set_seed(SEED)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
