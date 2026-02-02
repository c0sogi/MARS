import os
import random
import numpy as np
import torch


class Config:
    """
    Central configuration for the Coordinate Selective Kernel ResUNet (CSK-ResUNet) pipeline.
    Stores all hyperparameters, file paths, and execution settings.
    """

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "csk_resunet_best.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Cache Directory for data processing
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ==========================================
    # Data Configuration
    # ==========================================
    PATCH_SIZE = 128
    PATCHES_PER_IMAGE = 100  # High-density sampling: 100 patches per image per epoch

    # DataLoader settings
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # ==========================================
    # Model Architecture Configuration
    # ==========================================
    BASE_FILTERS = 64  # Base capacity for the network
    IN_CHANNELS = 1  # Grayscale input
    OUT_CHANNELS = 1  # Single channel output (Noise Residual)

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    NUM_EPOCHS = 100
    BATCH_SIZE = 32  # Optimized for A100 GPU and model complexity

    # Optimization
    LEARNING_RATE = 1e-3  # Initial learning rate
    WEIGHT_DECAY = 1e-2  # Strong regularization for SK blocks
    EARLY_STOPPING_PATIENCE = 15

    # Debugging flags
    DEBUG = False  # Set to True to train on a small subset
    DEBUG_SUBSET_SIZE = 10

    # ==========================================
    # Inference Configuration
    # ==========================================
    TILE_SIZE = 128
    TILE_OVERLAP = 0.5  # 50% overlap for sliding window inference
    USE_TTA = True  # Enable Test-Time Augmentation (Flip/Rotate ensemble)

    @staticmethod
    def set_seed(seed=42):
        """
        Sets fixed random seeds for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior in cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Global random seed set to {seed}")
