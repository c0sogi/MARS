import os
import torch


class Config:
    """
    Centralized configuration for the Ship vs Iceberg classification task.
    Implements settings for the Calibrated Cosine-SWA ResNet-18 Ensemble.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for the specific idea implementation
    WORKING_DIR = "./working/idea_26"

    # Sub-directories for artifacts
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Input Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Processing & Normalization
    # =========================================================================
    # Global statistics derived from data analysis for Independent Band Normalization
    # Band 1 (HH)
    BAND1_MIN = -45.5944
    BAND1_MAX = 32.1806

    # Band 2 (HV)
    BAND2_MIN = -45.6555
    BAND2_MAX = 17.8628

    # Image Dimensions
    ORIG_IMG_SIZE = 75
    IMG_SIZE = 224  # Upsampled size for ResNet-18
    CHANNELS = 3  # Band 1 (Norm), Band 2 (Norm), Average

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    NUM_CLASSES = 1
    DROPOUT_RATE = 0.5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    BATCH_SIZE = 64

    # Optimization (AdamW)
    LR_INIT = 2e-4
    WEIGHT_DECAY = 0.01

    # Loss Function
    LABEL_SMOOTHING = 0.05

    # Phase 1: Adaptive Calibration
    # Uses ReduceLROnPlateau to find natural convergence
    P1_MAX_EPOCHS = 100  # High ceiling, relies on Early Stopping
    P1_PATIENCE = 10
    P1_FACTOR = 0.5
    P1_MIN_LR = 1e-6

    # Phase 2: Production (Cosine-SWA)
    # Uses CosineAnnealingLR mapped to the calibrated epoch count
    SWA_LR = 1e-5
    SWA_EPOCHS = 12  # Fixed duration for SWA phase

    # Cross-Validation
    N_FOLDS = 5

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
