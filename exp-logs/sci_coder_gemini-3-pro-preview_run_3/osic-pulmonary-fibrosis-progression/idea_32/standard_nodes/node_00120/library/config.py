import os
import torch


class Config:
    """
    Configuration for the OSPR-Net (Output-Space Probabilistic Residual Network) task.
    Defines global hyperparameters, file paths, and constants.
    """

    # =========================================================================
    # File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directories
    WORKING_DIR = "./working/idea_32"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    # Image Preprocessing
    IMG_SIZE = 260
    SLICE_COUNT = 3  # Anchor slice + 2 boundary slices (Top/Bottom 50% area)
    IN_CHANNELS = 3  # 3 slices stacked as RGB channels

    # Radiological Windowing (Lung Window)
    WINDOW_LEVEL = -600
    WINDOW_WIDTH = 1500

    # Normalization Statistics (Derived from EDA)
    # Target FVC (Z-score standardization)
    TARGET_MEAN = 2654.6528
    TARGET_STD = 801.7017

    # Input Scalars
    AGE_MEAN = 67.5825
    AGE_STD = 6.6259

    # =========================================================================
    # Model Parameters
    # =========================================================================
    BACKBONE = "efficientnet_b2"
    PRETRAINED = True
    DROP_RATE = 0.2

    # =========================================================================
    # Training Parameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 50

    # Optimization
    LR_BACKBONE = 1e-4
    LR_HEAD = 1e-3
    WEIGHT_DECAY = 1e-2
    T_MAX = 50  # For Cosine Annealing

    # =========================================================================
    # Inference Parameters
    # =========================================================================
    CONFIDENCE_CLIP = 70  # Lower bound for sigma in submission

    # =========================================================================
    # System
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for caching and checkpoints.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories initialized at {cls.WORKING_DIR}")
