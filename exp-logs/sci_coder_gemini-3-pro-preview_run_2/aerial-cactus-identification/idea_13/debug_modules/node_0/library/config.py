import os


class Config:
    """
    Central configuration for Idea 13: Custom Narrow SE-ResNet with
    Multi-Scale Global Covariance Pooling.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"

    # Metadata Paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Directory for processed data (if needed)
    CACHE_DIR = WORKING_DIR

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Native resolution is 32x32. We strictly avoid resizing to prevent artifacts.
    IMAGE_SIZE = 32
    IN_CHANNELS = 3

    # Dataloader settings
    BATCH_SIZE = 64
    NUM_WORKERS = 2

    # Augmentation: Light augmentation only (Flips)
    USE_AUGMENTATION = True
    H_FLIP_PROB = 0.5
    V_FLIP_PROB = 0.5

    # =========================================================================
    # Model Architecture Configuration
    # =========================================================================
    # Custom Narrow SE-ResNet with Multi-Scale Global Covariance Pooling
    MODEL_NAME = "NarrowSEResNet_GCP"

    # Channel configuration for the 3 stages (Narrow width)
    # Reducing capacity to [16, 32, 64] as proven sufficient and efficient
    BLOCK_CHANNELS = [16, 32, 64]

    # Use Squeeze-and-Excitation blocks
    USE_SE = True
    SE_REDUCTION = 4  # Reduction ratio for SE blocks

    # Pooling Strategy: Global Covariance Pooling
    POOLING_TYPE = "covariance"  # Options: 'avg', 'max', 'gem', 'covariance'

    # =========================================================================
    # Training Configuration
    # =========================================================================
    # Homogeneous Seed Averaging Strategy
    SEEDS = [0, 1, 2, 3, 4]

    # Training duration
    EPOCHS = 15

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5

    # =========================================================================
    # Inference Configuration
    # =========================================================================
    # Test Time Augmentation (TTA)
    USE_TTA = True

    @classmethod
    def setup(cls):
        """
        Ensures that necessary working directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
