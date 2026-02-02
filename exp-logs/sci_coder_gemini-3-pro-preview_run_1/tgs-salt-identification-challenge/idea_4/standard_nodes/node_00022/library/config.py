import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
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


class Config:
    # -------------------------------------------------------------------------
    # 1. Global Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 8  # Optimized for 12 vCPUs

    # -------------------------------------------------------------------------
    # 2. File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata CSVs (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Experiment Directory (Idea 4)
    WORKING_DIR = "./working/idea_4"
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # 3. Data & Image Parameters
    # -------------------------------------------------------------------------
    ORIG_SIZE = 101
    IMG_SIZE = 128  # Padded size (divisible by 32 for U-Net)
    CHANNELS = 1  # Grayscale seismic images

    # -------------------------------------------------------------------------
    # 4. Model Hyperparameters (Residual U-Net)
    # -------------------------------------------------------------------------
    ENCODER_FILTERS = [64, 128, 256, 512, 1024]  # Filters at each stage
    DEEP_SUPERVISION = True
    DS_WEIGHTS = [1.0, 0.5, 0.25]  # Weights for [High-Res, Med-Res, Low-Res]

    # -------------------------------------------------------------------------
    # 5. Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 96  # Increased for A100 stability
    EPOCHS = 150  # Extended for full convergence (Cite Lesson 00011)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler Params (CosineAnnealingWarmRestarts)
    # T_0=10, T_mult=2 -> Restarts at epochs 10, 30, 70, 150
    T_0 = 10
    T_MULT = 2
    MIN_LR = 1e-6

    # -------------------------------------------------------------------------
    # 6. Debugging / Development Control
    # -------------------------------------------------------------------------
    # Toggle DEBUG to True to run on a small subset of data
    DEBUG = False
    DEBUG_SIZE = 100  # Number of samples to use when DEBUG is True


# Ensure necessary directories exist
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

# Apply seeding immediately upon import
seed_everything(Config.SEED)
