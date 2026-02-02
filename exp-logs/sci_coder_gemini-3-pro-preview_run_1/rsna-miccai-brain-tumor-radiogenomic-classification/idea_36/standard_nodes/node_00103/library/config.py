import os
import torch
import random
import numpy as np

# ==========================================
# Path Configurations
# ==========================================
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Working directory for caching processed data and saving models
WORKING_DIR = "./working"
IDEA_DIR = os.path.join(WORKING_DIR, "idea_36")
CACHE_DIR = IDEA_DIR
MODEL_SAVE_PATH = os.path.join(IDEA_DIR, "best_model.pth")
SUBMISSION_PATH = "./submission/submission.csv"

# Ensure working directories exist
os.makedirs(IDEA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# ==========================================
# Model Hyperparameters
# ==========================================
IMG_SIZE = 224
IN_CHANNELS = 9  # 3 modalities (FLAIR, T1wCE, T2w) * 3 depths (CoM-10%, CoM, CoM+10%)
BACKBONE = "tf_efficientnet_b0_ns"
NUM_CLASSES = 1
DROPOUT_RATE = 0.3

# ==========================================
# Data Processing Configuration
# ==========================================
# Relative offset from Center of Mass (CoM) for volumetric sampling
# 0.1 means +/- 10% of the brain ROI depth
DEPTH_OFFSET = 0.1

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
N_FOLDS = 5
SEED = 42
NUM_WORKERS = 4  # Optimized for 12 vCPUs

# Early Stopping
PATIENCE = 5
MIN_DELTA = 0.0001

# ==========================================
# System & Reproducibility
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
