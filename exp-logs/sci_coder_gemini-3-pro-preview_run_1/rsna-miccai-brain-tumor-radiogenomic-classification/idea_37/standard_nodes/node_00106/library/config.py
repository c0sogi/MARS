import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache directory for this idea to store processed tensors/stats
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_37")

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output paths
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    SEED = 42
    IMAGE_SIZE = (224, 224)

    # Modalities selected for the 9-channel stack (T1w excluded per design)
    MODALITIES = ["flair", "t1wce", "t2w"]

    # Volumetric Sampling Strategy
    # Offsets relative to Brain Depth (0.0 = Center of Mass)
    RELATIVE_OFFSETS = [-0.1, 0.0, 0.1]

    # Input Tensor Dimensions
    # 3 Modalities * 3 Slices = 9 Channels
    NUM_CHANNELS = len(MODALITIES) * len(RELATIVE_OFFSETS)

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True
    DROPOUT_RATE = 0.3

    # Gaussian Weight Inflation Initialization Factors
    # Center slices get 50% energy, peripheral slices get 25%
    WEIGHT_INFLATION_CENTER = 0.5
    WEIGHT_INFLATION_PERIPHERY = 0.25

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    NUM_FOLDS = 5

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # ==========================================
    # Augmentation Policy
    # ==========================================
    # Content-Based Alignment requires strict spatial anchoring
    AUG_ELASTIC = True
    AUG_GRID = True
    AUG_ROTATION = True
    AUG_FLIP = True

    # Strictly Excluded to preserve CoM alignment
    AUG_TRANSLATE = False
    AUG_SCALE = False

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Configuration setup complete. Device: {cls.DEVICE}")
        print(f"Cache Directory: {cls.CACHE_DIR}")
