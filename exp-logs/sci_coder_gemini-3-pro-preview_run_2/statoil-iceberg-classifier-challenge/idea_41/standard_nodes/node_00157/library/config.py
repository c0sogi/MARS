import os
import torch


class Config:
    """
    Global configuration for the Normalized-Fusion Wide-Body Network (NF-WBN) solution.
    """

    # ==========================================
    # PATHS
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for artifacts (models, cache)
    WORK_DIR = "./working/idea_41"

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # DATA SPECIFICATIONS
    # ==========================================
    IMG_HEIGHT = 75
    IMG_WIDTH = 75

    # Channels: Band_1, Band_2, (Band_1 + Band_2) / 2
    IN_CHANNELS = 3

    # Target Column
    TARGET_COL = "is_iceberg"

    # ==========================================
    # MODEL HYPERPARAMETERS (NF-WBN)
    # ==========================================
    # Sustained Width Strategy: Maintain 128 filters throughout backbone
    MODEL_FILTERS = 128

    # Regularization: High Dropout as per Lesson 77/14
    DROPOUT_RATE = 0.5

    # Metadata embedding dimension before fusion
    META_EMBED_DIM = 32

    # ==========================================
    # TRAINING HYPERPARAMETERS
    # ==========================================
    SEED = 42
    NUM_FOLDS = 5
    BATCH_SIZE = 32
    NUM_EPOCHS = 100

    # Optimizer settings (Adam)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0  # Only introduce if val loss > train loss

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 15

    # ==========================================
    # COMPUTE
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available, leave some overhead
    NUM_WORKERS = 4

    # ==========================================
    # DEBUGGING
    # ==========================================
    # Set to True to run on a small subset of data
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100

    @classmethod
    def setup(cls):
        """
        Initialize necessary directories.
        """
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Configuration loaded. Work Dir: {cls.WORK_DIR}, Device: {cls.DEVICE}")


# Automatically create directories on import
Config.setup()
