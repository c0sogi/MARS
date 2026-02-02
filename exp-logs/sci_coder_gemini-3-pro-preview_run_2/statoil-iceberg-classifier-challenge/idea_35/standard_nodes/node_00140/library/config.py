import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    PROJECT_NAME = "idea_35_ri_wbn"
    SEED = 42

    # Debugging flags to control dataset size
    DEBUG = False
    MAX_SAMPLES = None  # If DEBUG is True, limits dataset size (e.g., 100)

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_35"
    SUBMISSION_DIR = "./submission"

    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata (Pre-generated CSVs)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Artifacts
    # Cache file for processed tensors to avoid re-parsing JSONs
    PROCESSED_DATA_CACHE = os.path.join(WORKING_DIR, "processed_data.npz")
    # Output submission file
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Parameters
    # -------------------------------------------------------------------------
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    # Input Channels: Band 1 (HH), Band 2 (HV), Mean ((HH+HV)/2)
    IN_CHANNELS = 3
    NUM_CLASSES = 1  # Binary classification (0=Ship, 1=Iceberg)

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # Architecture specifics for RI-WBN
    DROPOUT_RATE = 0.5

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    NUM_FOLDS = 5
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3

    # Early Stopping
    PATIENCE = 15

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.1
    SCHEDULER_PATIENCE = 5

    # Regularization
    # Note: Weight decay is set to 0.0 as we rely on high dropout and augmentation
    WEIGHT_DECAY = 0.0

    # -------------------------------------------------------------------------
    # Hardware & System
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available, 4 workers is a safe balance for data loading
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories if they do not exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
