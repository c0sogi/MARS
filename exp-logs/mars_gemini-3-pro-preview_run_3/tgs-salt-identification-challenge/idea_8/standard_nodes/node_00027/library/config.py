import os
import random
import numpy as np
import torch


class Config:
    # ==============================
    # File Paths
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Specific working directory for this idea iteration
    WORKING_DIR = "./working/idea_9"

    # Metadata paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    DEPTHS_PATH = os.path.join(INPUT_DIR, "depths.csv")

    # Output paths
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==============================
    # Data Configuration
    # ==============================
    ORIG_SIZE = 101
    IMG_SIZE = 128  # Pad to power of 2 for U-Net
    CHANNELS = 3  # [Seismic, Seismic, Depth]
    N_FOLDS = 5
    NUM_WORKERS = 2  # Adjusted for available vCPUs

    # ==============================
    # Model Configuration
    # ==============================
    ENCODER = "resnext50_32x4d"
    ENCODER_WEIGHTS = "imagenet"
    DECODER_CHANNELS = (256, 128, 64, 32, 16)

    # ==============================
    # Training Configuration
    # ==============================
    SEED = 42
    BATCH_SIZE = 64
    EPOCHS = 80
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Curriculum Learning
    LOVASZ_SWITCH_EPOCH = 15
    EARLY_STOPPING_PATIENCE = 15

    # TTA
    TTA_STEPS = 1  # 1 means original + flip (2 total inferences per model)

    @staticmethod
    def setup():
        """
        Sets up the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        torch.cuda.manual_seed(Config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = (
            False  # False for exact reproducibility, True for speed
        )

        # Set device
        Config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Configuration setup complete.")
        print(f"Working Directory: {Config.WORKING_DIR}")
        print(f"Device: {Config.DEVICE}")
