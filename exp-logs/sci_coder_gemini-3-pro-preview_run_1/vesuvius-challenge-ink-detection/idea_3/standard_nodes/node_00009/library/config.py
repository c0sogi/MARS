import os
import random
import numpy as np
import torch


class Config:
    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = "./submission.csv"
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    PREDICTIONS_DIR = os.path.join(WORKING_DIR, "predictions")
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    # --- Data Configuration ---
    # Input volume depth (number of slices)
    Z_DIM = 65
    # Patch dimensions (Height, Width)
    PATCH_SIZE = (512, 512)
    # Number of workers for data loading
    NUM_WORKERS = 2

    # --- Model Configuration ---
    # Input channels (1 for grayscale x-ray)
    IN_CHANNELS = 1
    # Output channels (1 for binary ink detection)
    OUT_CHANNELS = 1
    # Model architecture specific params
    ENCODER_NAME = "resnet34"

    # --- Training Configuration ---
    # Explicitly requested batch size
    BATCH_SIZE = 1
    # Gradient accumulation steps
    GRAD_ACCUM_STEPS = 4
    # Learning rate
    LEARNING_RATE = 1e-4
    # Number of training epochs
    NUM_EPOCHS = 15
    # Positive class weight for BCE loss to handle imbalance
    POS_WEIGHT = 2.0
    # Early stopping patience
    PATIENCE = 5

    # --- Inference Configuration ---
    # Threshold search range for F0.5 optimization
    THRESHOLD_SEARCH_START = 0.2
    THRESHOLD_SEARCH_END = 0.8
    THRESHOLD_SEARCH_STEP = 0.05

    # --- Reproducibility ---
    SEED = 42

    # Compute device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
