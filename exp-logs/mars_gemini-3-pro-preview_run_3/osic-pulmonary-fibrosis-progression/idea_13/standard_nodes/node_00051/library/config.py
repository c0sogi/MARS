import os
import torch


class Config:
    """
    Configuration for the Dual-Stream Point-Wise Residual Network (DSPR-Net) pipeline.
    This module defines all hyperparameters, file paths, and constants.
    """

    # --------------------------------------------------------------------------
    # Reproducibility & Hardware
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Adjust number of workers based on available vCPUs (12)
    NUM_WORKERS = 4

    # --------------------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files (Pre-split by patient to prevent leakage)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directories for Idea 13
    WORKING_DIR = "./working/idea_13"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # --------------------------------------------------------------------------
    SLICES_PER_PATIENT = 3
    IMG_SIZE = 260  # Native resolution for EfficientNet-B2

    # Feature Engineering
    # Scale relative weeks to keep values small (e.g., 100 weeks -> 1.0)
    WEEKS_SCALE = 0.01

    # Debugging / Development
    # Set MAX_TRAIN_SAMPLES to an integer (e.g., 100) to debug on a small subset
    DEBUG = False
    MAX_TRAIN_SAMPLES = None

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # --------------------------------------------------------------------------
    BACKBONE = "efficientnet_b2"
    PRETRAINED = True

    # Image Branch
    IMAGE_EMBEDDING_DIM = 256

    # Tabular Branch
    # Static features input to the Deep Stream (Stream A)
    # Note: 'Weeks' is transformed to t_rel. 'FVC' is the target.
    TABULAR_FEATURES = ["Age", "Sex", "SmokingStatus", "Baseline_FVC"]

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    EPOCHS = 50
    BATCH_SIZE = 32

    # Differential Learning Rates
    # Lower LR for backbone to preserve pretrained features
    LR_BACKBONE = 1e-4
    # Higher LR for heads to learn task-specific projections
    LR_HEAD = 1e-3

    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = 50
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 10

    # Metric Constraints
    CONFIDENCE_CLIP = 70.0
    ERROR_THRESHOLD = 1000.0

    @classmethod
    def create_directories(cls):
        """Creates the necessary directories for the experiment."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically create directories when config is imported
Config.create_directories()
