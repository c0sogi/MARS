import os
import torch


class Config:
    """
    Centralized configuration for the Deep Pre-Activation Parallel DCN-ResNet task.
    Includes file paths, hyperparameters, and system settings.
    """

    # -------------------------------------------------------------------------
    # System & Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42
    # Use GPU if available, otherwise CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Number of data loading workers (12 vCPUs available, 4 is a safe default)
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea iteration
    WORKING_DIR = "./working/idea_26"
    SUBMISSION_DIR = "./submission"

    # Input Data Paths (using generated metadata parquet files)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache Paths (for deterministic data processing)
    # Using .npy for efficient storage of processed numpy arrays
    CACHE_TRAIN_X = os.path.join(WORKING_DIR, "X_train.npy")
    CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "y_train.npy")
    CACHE_VAL_X = os.path.join(WORKING_DIR, "X_val.npy")
    CACHE_VAL_Y = os.path.join(WORKING_DIR, "y_val.npy")
    CACHE_TEST_X = os.path.join(WORKING_DIR, "X_test.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Deep Pre-Activation Parallel DCN-ResNet
    HIDDEN_DIM = 512
    NUM_RESNET_BLOCKS = 4  # Depth of the ResNet backbone
    DROPOUT_RATE = 0.2  # Regularization strength
    NUM_CLASSES = 7  # Target classes (Cover_Type 1-7)

    # Note: INPUT_DIM is determined dynamically after feature engineering

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 4096
    EPOCHS = 60

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.1
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MODE = "max"  # Monitoring accuracy

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    @classmethod
    def create_directories(cls):
        """
        Ensures that the working and submission directories exist.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
