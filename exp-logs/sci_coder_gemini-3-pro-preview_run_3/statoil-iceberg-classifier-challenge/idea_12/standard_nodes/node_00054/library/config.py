import os
import torch


class Config:
    """
    Central configuration for the Aggressively Downsampled SE-ResNet Ensemble.
    """

    # --------------------------------------------------------------------------
    # Directory and File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache Directory for processed numpy arrays
    CACHE_DIR = WORKING_DIR

    # Output Submission File
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Parameters
    # --------------------------------------------------------------------------
    IMG_SIZE = 75
    IN_CHANNELS = 3  # HH, HV, Avg((HH, HV))
    NUM_CLASSES = 1  # Binary Classification (Ship vs Iceberg)

    # Debugging: Set to an integer (e.g., 100) to limit dataset size for fast checking
    # Set to None for full training
    MAX_SAMPLES = None

    # --------------------------------------------------------------------------
    # Model Architecture Parameters
    # --------------------------------------------------------------------------
    # Architecture: SimpleCNN

    # --------------------------------------------------------------------------
    # Training Parameters
    # --------------------------------------------------------------------------
    SEED = 42
    NUM_FOLDS = 5

    # Optimization
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3  # Constant LR
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    # Early Stopping
    PATIENCE = 10

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 40)
        print(" CONFIGURATION")
        print("=" * 40)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k:<20}: {v}")
        print("=" * 40)
