import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
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


class Config:
    """
    Global configuration for the Technosignature Detection project.
    """

    # --- System Configuration ---
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for the 12 vCPU environment

    # --- File Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata paths (pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working directory for caching and model checkpoints
    WORKING_DIR = "./working/idea_3"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Model checkpoint path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "r3d_18_best.pth")

    # Submission directory and path
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Dimensions ---
    # The input is a 6-frame sequence of 273x256 spectrograms
    # Shape: (Batch, Channels, Depth, Height, Width) -> (B, 1, 6, 273, 256)
    IMG_HEIGHT = 273
    IMG_WIDTH = 256
    DEPTH = 6  # Number of cadence panels (ABACAD)
    IN_CHANNELS = 1  # Single channel intensity
    NUM_CLASSES = 1  # Binary classification

    # --- Training Hyperparameters ---
    DEBUG = False  # Set to True to train on a small subset
    DEBUG_SUBSET_SIZE = 2000  # Number of samples to use if DEBUG is True

    BATCH_SIZE = 32  # Fits comfortably in A100 40GB with R3D-18
    EPOCHS = 20  # Max epochs, controlled by Early Stopping
    LEARNING_RATE = 1e-3  # Base learning rate
    WEIGHT_DECAY = 1e-2  # For AdamW optimizer
    MAX_LR = 1e-2  # Max learning rate for OneCycleLR

    # Mixup
    MIXUP_ALPHA = 1.0
    MIXUP_PROB = 0.5

    # Early Stopping
    PATIENCE = 8  # Increased patience for longer training

    # --- Model Architecture ---
    MODEL_NAME = "efficientnet_b0"  # EfficientNet-B0 with 6 input channels


# Apply seed immediately
seed_everything(Config.SEED)
