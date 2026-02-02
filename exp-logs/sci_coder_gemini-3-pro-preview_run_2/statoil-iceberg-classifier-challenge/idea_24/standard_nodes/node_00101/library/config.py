import os
import torch


class Config:
    """
    Central configuration for the Post-Pooling Contracted Wide-Body Network (PPC-WBN) pipeline.
    """

    # ==========================================
    # 1. PATHS & DIRECTORIES
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for artifacts (checkpoints, cache)
    # Using 'idea_24' as the designated workspace
    WORKING_DIR = "./working/idea_24"

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Create directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Caching Path (Parquet or NPZ)
    # We use .npz for efficient numpy array storage of processed images
    CACHE_PATH = os.path.join(WORKING_DIR, "processed_data.npz")

    # ==========================================
    # 2. DATA CONFIGURATION
    # ==========================================
    IMG_HEIGHT = 75
    IMG_WIDTH = 75

    # Channels: Band 1, Band 2, and (Band 1 + Band 2) / 2
    IN_CHANNELS = 3

    # Augmentation Settings
    # Rotational invariance: 0, 90, 180, 270 degrees
    ROTATION_ANGLES = [0, 90, 180, 270]

    # ==========================================
    # 3. MODEL CONFIGURATION
    # ==========================================
    # Architecture: Post-Pooling Contracted Wide-Body Network
    BACKBONE_WIDTH = 128  # Sustained width
    DROPOUT_RATE = 0.5  # High dropout for regularization
    NUM_CLASSES = 1  # Binary classification (Sigmoid)

    # ==========================================
    # 4. TRAINING CONFIGURATION
    # ==========================================
    SEED = 42
    NUM_FOLDS = 5
    BATCH_SIZE = 32

    # Optimization
    LEARNING_RATE = 2e-4  # Conservative start ("Low and Slow")
    NUM_EPOCHS = 100  # Upper limit
    PATIENCE = 10  # Early stopping patience

    # Scheduler
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 4

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Adjust workers based on 12 vCPUs available
    NUM_WORKERS = 4

    # ==========================================
    # 5. DEBUGGING & CONTROL
    # ==========================================
    # Set to True to run on a small subset for testing pipeline mechanics
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print("PIPELINE CONFIGURATION")
        print("=" * 30)
        for key, val in cls.__dict__.items():
            if not key.startswith("__") and not callable(val):
                print(f"{key}: {val}")
        print("=" * 30)
