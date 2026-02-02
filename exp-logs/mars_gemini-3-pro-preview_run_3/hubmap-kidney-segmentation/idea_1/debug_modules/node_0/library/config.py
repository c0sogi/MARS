import os
import torch
import random
import numpy as np


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # ==========================
    # Paths
    # ==========================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Output directories
    # Using 'idea_1' as a specific workspace for this approach
    ARTIFACT_DIR = os.path.join(WORKING_DIR, "idea_1")
    SUBMISSION_DIR = "./submission"

    # Create directories if they don't exist
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Model Checkpoint and Submission
    MODEL_SAVE_PATH = os.path.join(ARTIFACT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================
    # Data Processing
    # ==========================
    TILE_SIZE = 1024
    MIN_OVERLAP = 128  # Minimum overlap in pixels for tiling
    # Stride is calculated as Tile Size - Overlap
    STRIDE = TILE_SIZE - MIN_OVERLAP

    # Number of workers for data loading
    NUM_WORKERS = 4

    # ==========================
    # Model Architecture
    # ==========================
    ENCODER = "resnet18"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 3
    CLASSES = 1  # Binary segmentation

    # ==========================
    # Training Hyperparameters
    # ==========================
    BATCH_SIZE = 16
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5

    # ==========================
    # Compute & Reproducibility
    # ==========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    # ==========================
    # Debugging
    # ==========================
    # If True, the pipeline will use a small subset of data for quick testing
    DEBUG = False
