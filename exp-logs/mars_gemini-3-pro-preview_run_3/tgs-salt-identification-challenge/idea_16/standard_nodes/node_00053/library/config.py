import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    PROJECT_NAME = "SaltSegmentation_Idea16"

    # =========================================================================
    # Paths
    # =========================================================================
    # Input paths (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working paths (Write Allowed)
    # Using idea_16 specific directory for isolation
    WORK_DIR = "./working/idea_16"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    LOG_DIR = os.path.join(WORK_DIR, "logs")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Original image size
    IMG_HEIGHT_ORIG = 101
    IMG_WIDTH_ORIG = 101

    # Training image size (Padded to power of 32 for U-Net)
    IMG_HEIGHT_TRAIN = 128
    IMG_WIDTH_TRAIN = 128

    # Input channels: 3 (Seismic, Seismic, Depth)
    IN_CHANNELS = 3

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_ARCH = "UnetPlusPlus"
    ENCODER_NAME = "seresnext50_32x4d"
    ENCODER_WEIGHTS = "imagenet"

    # Lightweight decoder channels to prevent overfitting
    # Starting at 16 filters in the final layer
    DECODER_CHANNELS = (256, 128, 64, 32, 16)

    # Attention mechanism
    DECODER_ATTENTION_TYPE = "scse"

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    N_FOLDS = 5
    EPOCHS = 80

    # Phase 1: Structural Warm-up (BCE+Dice, Deep Supervision Active)
    # Runs from Epoch 1 to PHASE1_EPOCHS
    PHASE1_EPOCHS = 20

    # Phase 2: Metric Fine-tuning (Lovasz, Deep Supervision Disabled)
    # Runs from PHASE1_EPOCHS + 1 to EPOCHS

    BATCH_SIZE = 64

    # Learning Rates
    LR_MAX = 1e-4
    LR_MIN = 1e-6

    # Optimizer
    WEIGHT_DECAY = 1e-4

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5

    # =========================================================================
    # Compute
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    @classmethod
    def setup(cls):
        """Creates necessary directories for the experiment."""
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
