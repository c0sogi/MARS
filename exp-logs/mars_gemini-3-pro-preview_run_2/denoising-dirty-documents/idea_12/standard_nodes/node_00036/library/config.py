import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Coordinate Sub-Pixel ResUNet (CoSP-ResUNet) task.
    Centralizes all hyperparameters, file paths, and system settings.
    """

    # --- Project Structure & Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"

    # Input Data Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "cosp_resunet_best.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Configuration ---
    # Patch size for training (strictly 128x128 as per strategy)
    PATCH_SIZE = 128
    # High-density sampling: Number of random patches extracted per image per epoch
    PATCHES_PER_EPOCH = 100
    # Input channels (Grayscale)
    NUM_CHANNELS = 1

    # --- Model Configuration ---
    # Base filter capacity for the U-Net
    BASE_FILTERS = 64

    # --- Training Configuration ---
    SEED = 42
    EPOCHS = 100
    # Batch size adapted for A100 GPU and 128x128 patches
    BATCH_SIZE = 32
    # Initial learning rate for AdamW
    LEARNING_RATE = 1e-3
    # Strong weight decay for regularization
    WEIGHT_DECAY = 1e-2
    # Early stopping patience
    EARLY_STOPPING_PATIENCE = 15
    # Number of data loading workers
    NUM_WORKERS = 4

    # --- Inference Configuration ---
    # Overlap ratio for tiled inference (50% overlap)
    TILE_OVERLAP = 0.5

    # --- System Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def initialize(cls):
        """
        Performs necessary setup: creates directories and sets random seeds
        for reproducible execution.
        """
        # Create necessary directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set fixed random seeds for reproducibility
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior for CuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
