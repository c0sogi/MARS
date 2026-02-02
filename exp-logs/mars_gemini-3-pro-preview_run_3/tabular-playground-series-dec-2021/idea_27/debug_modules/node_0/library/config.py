import os
import torch


class Config:
    """
    Central configuration for the Deep Parallel Vector-DCN-ResNet pipeline.
    Defines hyperparameters, file paths, and compute settings.
    """

    # ==========================================
    # Global Seeding
    # ==========================================
    SEED = 42

    # ==========================================
    # Data Paths
    # ==========================================
    # Input Metadata (Parquet files)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output Directories
    # working/idea_27 is used for caching processed data and model checkpoints
    WORKING_DIR = "./working/idea_27"
    # submission/ is used for the final CSV output
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Specific Output Files
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    # Architecture: Deep Parallel Vector-DCN-ResNet (Full Pre-Activation)
    HIDDEN_DIM = 512  # Width of hidden layers
    NUM_BLOCKS = 4  # Number of ResNet blocks
    DROPOUT = 0.2  # Dropout rate for regularization

    # Input dimensions (will be determined dynamically, but constants can be set here if fixed)
    # The model handles continuous and categorical features separately.

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 4096
    EPOCHS = 60
    LEARNING_RATE = 1e-3  # Initial learning rate for AdamW
    WEIGHT_DECAY = 1e-2  # Weight decay for AdamW

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.1
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MODE = "max"  # Monitor accuracy (max)

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 12

    # ==========================================
    # Compute Settings
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available, leaving some overhead
    NUM_WORKERS = 4
