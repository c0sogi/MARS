import os
import torch
import random
import numpy as np


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Config:
    """
    Central configuration class for the Interleaved Slice-Grouped 2.5D Network.
    Defines paths, hyperparameters, and model settings.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata File Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Caching Directory (Idea 15)
    CACHE_DIR = "./working/idea_15"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    NUM_SLICES = 32  # High-density uniform sampling
    IMG_SIZE = 256  # Spatial resolution (H, W)
    NUM_MODALITIES = 4  # FLAIR, T1w, T1wCE, T2w

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    # Input tensor shape: (B, 128, 256, 256)
    # 128 channels = 32 slices * 4 modalities
    IN_CHANNELS = NUM_SLICES * NUM_MODALITIES

    # Stem Configuration
    STEM_OUT_CHANNELS = 128
    STEM_GROUPS = 32  # 1 group per slice depth (processing 4 channels/group)
    BACKBONE_IN_CHANNELS = 64  # Output channels after depth aggregation

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 16  # Adjusted for A100 GPU memory
    LEARNING_RATE = 1e-4  # Low LR for stability
    EPOCHS = 15  # Max epochs
    EARLY_STOPPING_PATIENCE = 3

    # ==========================================
    # Compute & Debugging
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging
    DEBUG = False  # Toggle to use a smaller dataset subset
    DEBUG_SIZE = 50  # Number of samples in debug mode
