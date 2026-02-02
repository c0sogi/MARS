import os
import random
import numpy as np
import torch


class Config:
    # --------------------------------------------------------------------------
    # Directory & File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory specific to this idea iteration
    WORK_DIR = "./working/idea_45"
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORK_DIR, "cache")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files
    TRAIN_META_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_CSV = os.path.join(METADATA_DIR, "test.csv")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    IMG_SIZE = 75
    # 3 Channels: Band 1 (HH), Band 2 (HV), Average ((HH+HV)/2)
    IN_CHANNELS = 3

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    # Architecture: Bottlenecked Dual-Polarity Hierarchical CNN
    MODEL_NAME = "BDPH_CNN"

    # Backbone widths (Plain CNN 4-stage)
    BACKBONE_CHANNELS = [64, 128, 128, 128]

    # Bottleneck readout dimension (128 -> 32 compression before pooling)
    BOTTLENECK_DIM = 32

    # Regularization
    DROPOUT_RATE = 0.5
    LEAKY_RELU_SLOPE = 0.1

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    N_FOLDS = 5

    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3  # Constant LR
    WEIGHT_DECAY = 0.01  # For AdamW

    EPOCHS = 75
    PATIENCE = 12  # Early stopping patience

    NUM_WORKERS = 2  # Data loading workers

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Sets up the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set random seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Configuration setup complete. Working directory: {cls.WORK_DIR}")
        print(f"Device: {cls.DEVICE}")
