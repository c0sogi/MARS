import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    PROJECT_NAME = "MA-COSR-Lung-Decline"
    EXPERIMENT_ID = "idea_34"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    # Input Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Paths
    WORKING_DIR = os.path.join("./working", EXPERIMENT_ID)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Model Save Path
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Preprocessing
    # -------------------------------------------------------------------------
    # Image Parameters
    IMAGE_SIZE = 260  # EfficientNet-B2 native resolution
    SLICE_COUNT = 3  # Anchor + 2 boundary slices

    # DICOM Windowing (Lung Window)
    WINDOW_LEVEL = -600
    WINDOW_WIDTH = 1500

    # Normalization
    PIXEL_MEAN = 0.5  # Approximate mean after min-max scaling
    PIXEL_STD = 0.5  # Approximate std after min-max scaling

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE_NAME = "efficientnet_b2"
    PRETRAINED = True

    # Clinical Stream
    CLINICAL_INPUT_DIM = 4  # Age, Sex, SmokingStatus, RelativeTime
    CLINICAL_HIDDEN_DIM = 128
    CLINICAL_LATENT_DIM = 64

    # Fusion
    DROPOUT_RATE = 0.2

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    EPOCHS = 50
    BATCH_SIZE = 32

    # Optimization
    # Differential Learning Rates
    LR_BACKBONE = 1e-4  # Slower learning for visual features
    LR_HEADS = 1e-3  # Faster learning for clinical/fusion heads
    WEIGHT_DECAY = 1e-2

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # -------------------------------------------------------------------------
    # Metric / Loss
    # -------------------------------------------------------------------------
    # Modified Laplace Log Likelihood constants
    QUANTILE_CLIP = 70  # Lower bound for sigma (ml)
    MAX_ERROR = 1000  # Upper bound for absolute error (ml)

    @classmethod
    def mkdirs(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically create directories when module is imported/used
Config.mkdirs()
