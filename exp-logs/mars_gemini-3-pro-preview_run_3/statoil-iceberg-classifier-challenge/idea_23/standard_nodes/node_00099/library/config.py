import os
import torch


class Config:
    """
    Configuration for the Multi-Scale Max-Attention Network (MSMA-Net) solution.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_23"
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (Saved in WORKING_DIR)
    # These names are used by the data processing module
    CACHE_TRAIN_DATA = os.path.join(WORKING_DIR, "train_data.npy")
    CACHE_TEST_DATA = os.path.join(WORKING_DIR, "test_data.npy")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    # 3 Channels: HH, HV, and Average((HH+HV)/2)
    IN_CHANNELS = 3
    NUM_CLASSES = 1

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    MODEL_NAME = "MSMA-Net"
    # Dropout rate for the classification head
    DROPOUT_RATE = 0.5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    NUM_FOLDS = 5

    # Training Budget
    EPOCHS = 75
    PATIENCE = 12

    # Optimization
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    # =========================================================================
    # System & Debugging
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging flags to speed up development cycles if needed
    # If DEBUG is True, the data loader will only load MAX_DEBUG_SAMPLES
    DEBUG = False
    MAX_DEBUG_SAMPLES = 200

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("-" * 30)
        print("Configuration:")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Model: {cls.MODEL_NAME}")
        print(f"  Input Shape: ({cls.IN_CHANNELS}, {cls.IMG_HEIGHT}, {cls.IMG_WIDTH})")
        print(f"  Epochs: {cls.EPOCHS}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        print(f"  Dropout: {cls.DROPOUT_RATE}")
        print(f"  Working Dir: {cls.WORKING_DIR}")
        print("-" * 30)
