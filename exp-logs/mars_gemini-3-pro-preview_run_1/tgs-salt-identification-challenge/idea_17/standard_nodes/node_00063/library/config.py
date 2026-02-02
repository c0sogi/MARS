import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration for Deep Residual U-Net with Extended Homogeneous Lovasz-Ensemble.
    """

    # -------------------------------------------------------------------------
    # 1. Paths and Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_17"

    # Metadata files (Pre-generated 80/20 split)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output directories
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    PREDICTION_DIR = os.path.join(WORKING_DIR, "predictions")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # 2. Data and Image Parameters
    # -------------------------------------------------------------------------
    # Original image dimensions
    ORIG_H = 101
    ORIG_W = 101

    # Model input dimensions (Padded with reflection to nearest multiple of 32)
    IMG_H = 128
    IMG_W = 128

    # Input channels: 1 (Grayscale Image) + 1 (Depth Channel) = 2
    INPUT_CHANNELS = 2

    # -------------------------------------------------------------------------
    # 3. Model Parameters
    # -------------------------------------------------------------------------
    # Deep Residual Encoder filters (Avoiding 1024 to prevent convergence issues)
    ENCODER_FILTERS = [64, 128, 256, 512]

    # Enable Deep Supervision heads at resolutions 32, 64, 128
    DEEP_SUPERVISION = True

    # -------------------------------------------------------------------------
    # 4. Training Schedule & Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    BATCH_SIZE = 32  # Optimized for A100 40GB with 128x128 input

    # Optimizer (AdamW)
    LR_MAX = 1e-3
    LR_MIN = 1e-6
    WEIGHT_DECAY = 1e-4

    # Cyclic Schedule
    CYCLES = 4
    EPOCHS_PER_CYCLE = 50
    TOTAL_EPOCHS = CYCLES * EPOCHS_PER_CYCLE  # 200 Epochs

    # Loss Schedule Boundaries
    # Phase 1 (Epochs 1-50): BCE + Sample-Wise Dice
    # Phase 2 (Epochs 51-200): BCE + Lovasz-Hinge
    CYCLE_1_END_EPOCH = 50

    # -------------------------------------------------------------------------
    # 5. Ensembling & Inference
    # -------------------------------------------------------------------------
    # Quality Gate: Models within 0.5% mAP of the best model are included in ensemble
    QUALITY_GATE_THRESHOLD = 0.005

    # Test-Time Augmentation
    TTA_FLIP = True

    @staticmethod
    def setup_directories():
        """Ensure necessary working and submission directories exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.PREDICTION_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    @staticmethod
    def set_seed(seed=42):
        """Sets fixed random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
