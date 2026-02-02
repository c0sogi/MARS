import os
import random
import numpy as np
import torch


class Config:
    # ==========================================
    # 1. PATHS & DIRECTORIES
    # ==========================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea (Idea 22)
    # This is where checkpoints and cached data will be stored
    WORK_DIR = "./working/idea_22"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # 2. DATA SPECIFICATIONS
    # ==========================================
    IMG_HEIGHT = 75
    IMG_WIDTH = 75

    # Input Channels: 3 (Band 1, Band 2, Average)
    IN_CHANNELS = 3

    # Classes: Ship (0), Iceberg (1)
    NUM_CLASSES = 1

    # ==========================================
    # 3. TRAINING HYPERPARAMETERS
    # ==========================================
    # "Low and Slow" strategy
    LEARNING_RATE = 2e-4

    # Batch size
    BATCH_SIZE = 64

    # Training duration
    NUM_EPOCHS = 100

    # Early Stopping
    PATIENCE = 15

    # Cross Validation
    NUM_FOLDS = 5

    # ==========================================
    # 4. MODEL HYPERPARAMETERS
    # ==========================================
    # Convolutional filters for the 4 stages
    # Stage 4 contracts to 64 to avoid aggressive bottleneck (Cite solution_lesson_node_00055)
    FILTERS = [64, 128, 128, 64]

    # Dropout rate for the dense head
    # Increased to 0.5 to prevent overfitting with wider model (Cite solution_lesson_node_00077)
    DROPOUT_RATE = 0.5

    # ==========================================
    # 5. HARDWARE & REPRODUCIBILITY
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates necessary working directories for artifacts and cache.
        """
        dirs = [cls.WORK_DIR, cls.CACHE_DIR, cls.CHECKPOINT_DIR, cls.SUBMISSION_DIR]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        print(f"Configuration initialized. Working directory: {cls.WORK_DIR}")

    @classmethod
    def set_seed(cls, seed=None):
        """
        Sets fixed random seeds for reproducibility across libraries.
        """
        if seed is None:
            seed = cls.SEED

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior in cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"Random seed set to: {seed}")
