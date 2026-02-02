import os
import torch


class Config:
    """
    Central configuration for the Dual-Polarity DropBlock SE-CNN model.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SIZE = 100  # Number of samples to use in debug mode
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Input Data
    INPUT_DIR = "./input"
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working & Caching
    # Using specific directory for this idea as requested
    WORKING_DIR = "./working/idea_38"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Cache File Paths
    CACHE_X_TRAIN = os.path.join(CACHE_DIR, "X_train.npy")
    CACHE_Y_TRAIN = os.path.join(CACHE_DIR, "y_train.npy")
    CACHE_ANGLE_TRAIN = os.path.join(CACHE_DIR, "angle_train.npy")

    CACHE_X_VAL = os.path.join(CACHE_DIR, "X_val.npy")
    CACHE_Y_VAL = os.path.join(CACHE_DIR, "y_val.npy")
    CACHE_ANGLE_VAL = os.path.join(CACHE_DIR, "angle_val.npy")

    CACHE_X_TEST = os.path.join(CACHE_DIR, "X_test.npy")
    CACHE_ANGLE_TEST = os.path.join(CACHE_DIR, "angle_test.npy")
    CACHE_ID_TEST = os.path.join(CACHE_DIR, "id_test.npy")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Model Architecture Hyperparameters
    # =========================================================================
    # Input dimensions
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    IN_CHANNELS = 3  # Band 1 (HH), Band 2 (HV), Average ((HH+HV)/2)

    # Backbone (Plain CNN)
    # Channel expansion strategy: 64 -> 128 -> 128 -> 128
    LAYER_CHANNELS = [64, 128, 128, 128]

    # Activation
    LEAKY_RELU_SLOPE = 0.1  # Preserves shadow information (negative values)

    # Regularization
    DROPBLOCK_PROB_MAX = 0.1  # Max probability for DropBlock schedule
    DROPBLOCK_BLOCK_SIZE = 3  # Size of blocks to drop (3x3)
    CLASSIFIER_DROPOUT = 0.5  # Dropout rate in the classification head

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    NUM_FOLDS = 5
    NUM_EPOCHS = 75
    BATCH_SIZE = 32

    # Optimization
    LEARNING_RATE = 1e-3  # Constant LR
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    # Early Stopping
    PATIENCE = 12
    EARLY_STOPPING_DELTA = 1e-4

    # =========================================================================
    # Utilities
    # =========================================================================
    @classmethod
    def print_summary(cls):
        """Prints a summary of the configuration."""
        print("-" * 40)
        print("CONFIG SUMMARY")
        print("-" * 40)
        print(f"Device: {cls.DEVICE}")
        print(f"Seed: {cls.SEED}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.NUM_EPOCHS} (Patience: {cls.PATIENCE})")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"DropBlock Max Prob: {cls.DROPBLOCK_PROB_MAX}")
        print(f"Working Dir: {cls.WORKING_DIR}")
        print("-" * 40)
