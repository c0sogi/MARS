import os
import random
import numpy as np
import torch


class Config:
    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORK_DIR = "./working/idea_31"

    # Metadata Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data Paths
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Output Paths
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    @staticmethod
    def set_seed(seed=42):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    @staticmethod
    def setup_dirs():
        """Creates necessary directories."""
        os.makedirs(Config.WORK_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Hyperparameters
    # =========================================================================
    IMG_SIZE = 224  # Upsampled size for ResNet
    BATCH_SIZE = 32
    NUM_WORKERS = 2

    # Normalization Constants (Derived from Data Analysis)
    # Band 1 (HH)
    BAND1_MIN = -45.5944
    BAND1_MAX = 32.1806

    # Band 2 (HV)
    BAND2_MIN = -45.6555
    BAND2_MAX = 17.8628

    # Incidence Angle
    INC_ANGLE_MEAN = 39.2829
    INC_ANGLE_STD = 3.8362

    # Missing Incidence Angle Imputation Value
    INC_ANGLE_FILL = INC_ANGLE_MEAN

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    BACKBONE = "resnet18"
    PRETRAINED = True
    DROPOUT_RATE = 0.5
    NUM_CLASSES = 1

    # Input Channels: 3 (Band 1 Norm, Band 2 Norm, Average of B1+B2)
    IN_CHANNELS = 3

    # =========================================================================
    # Training Protocol - Phase 1: Calibration (CV)
    # =========================================================================
    # Optimizer
    LR_BASE = 2e-4
    WEIGHT_DECAY = 0.01

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 10
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 25
    MAX_EPOCHS_PHASE1 = 100  # Upper bound for calibration

    # =========================================================================
    # Training Protocol - Phase 2: Production (Isovariant Full-Fit)
    # =========================================================================
    # Isovariant Scaling Factor:
    # Production Epochs = ceil(Optimal_CV_Epochs * ISOVARIANT_SCALE)
    # Scale = (Train_Size_CV / Train_Size_Full) ~= 0.8
    ISOVARIANT_SCALE = 0.8

    # SWA Configuration
    LR_SWA = 1e-5
    SWA_EPOCHS = 12

    # Loss Function
    LABEL_SMOOTHING = 0.05

    # =========================================================================
    # Inference
    # =========================================================================
    TTA_STEPS = 4  # Original, H-Flip, V-Flip, Rotate180
