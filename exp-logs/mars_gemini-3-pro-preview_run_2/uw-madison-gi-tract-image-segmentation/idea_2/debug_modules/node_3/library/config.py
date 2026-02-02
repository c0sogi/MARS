import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the Lightweight SegFormer MRI Segmentation project.
    """

    # ==========================
    # File Paths & Directories
    # ==========================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching and model checkpoints
    WORKING_DIR = "./working/idea_2"
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata CSV paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    if not os.path.exists(SUBMISSION_DIR):
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================
    # Data Configuration
    # ==========================
    IMAGE_SIZE = 256  # Resolution for MiT-B0
    IN_CHANNELS = 3  # 2.5D Input: Slice i-1, Slice i, Slice i+1
    NUM_CLASSES = 3  # Large Bowel, Small Bowel, Stomach

    # Normalization parameters (if using standard mean/std, otherwise min-max is used in code)
    # These are placeholders if specific normalization is needed later
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]

    # ==========================
    # Model Configuration
    # ==========================
    BACKBONE = "mit_b0"  # Mix Transformer B0
    DECODER = "MLP"  # All-MLP Decoder

    # ==========================
    # Training Hyperparameters
    # ==========================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # For AdamW

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # Adjusted for 12 vCPUs availability

    # Debugging
    DEBUG = False  # Set True to run on a small subset of data


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
