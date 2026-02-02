import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # File System Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output & Caching Paths
    # We use idea_25 as the specific working directory for this experiment
    WORK_DIR = "./working/idea_25"
    CACHE_DIR = WORK_DIR
    MODEL_SAVE_PATH = os.path.join(WORK_DIR, "best_model.pth")

    # Submission Path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    # S3HD Network Requirement: 32 slices per modality
    SLICES_PER_MODALITY = 32
    NUM_MODALITIES = 4

    # Input Shape: (B, 128, 224, 224)
    # 128 channels = 32 slices * 4 modalities
    INPUT_CHANNELS = SLICES_PER_MODALITY * NUM_MODALITIES
    IMAGE_SIZE = 224
    INPUT_SHAPE = (INPUT_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

    # ==========================================
    # Model Configuration
    # ==========================================
    BACKBONE_NAME = "efficientnet_b0"
    # Regularization for small dataset generalization
    DROP_PATH_RATE = 0.2
    NUM_CLASSES = 1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 16  # Adjusted for 128-channel input on A100
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 15

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # 1. Create Directories
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # 2. Set Random Seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Configuration initialized. Device: {cls.DEVICE}, Seed: {cls.SEED}")
        print(f"Working Directory: {cls.WORK_DIR}")
