import os
import torch
import random
import numpy as np


class Config:
    """
    Centralized configuration for the Asymmetric Grouped EfficientNet pipeline.
    Includes file paths, data processing parameters, and training hyperparameters.
    """

    # -------------------------------------------------------------------------
    # File System & Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Specific working directory for Idea 20 (Caching & Checkpoints)
    WORKING_DIR = "./working/idea_20"

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    # Modalities to use (Order matters for channel stacking)
    MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]

    # Image Dimensions
    IMG_SIZE = (224, 224)

    # ROI Selection / Stacking
    NUM_SLICES_PER_MODALITY = 3  # Center slice +/- neighbors
    TOTAL_CHANNELS = len(MODALITIES) * NUM_SLICES_PER_MODALITY  # 4 * 3 = 12 channels
    STRIDE = 5  # Fixed stride for neighbor selection

    # Depth Search Bounds (Percentage of volume depth)
    ANCHOR_MIN_QUANTILE = 0.15
    ANCHOR_MAX_QUANTILE = 0.85

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True
    NUM_CLASSES = 1
    DROPOUT_RATE = 0.3

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    PATIENCE = 5  # For Early Stopping
    NUM_WORKERS = 4

    # Compute Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def setup_reproducibility(seed=42):
        """
        Sets fixed random seeds for Python, NumPy, and PyTorch to ensure
        reproducible results.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        os.environ["PYTHONHASHSEED"] = str(seed)
