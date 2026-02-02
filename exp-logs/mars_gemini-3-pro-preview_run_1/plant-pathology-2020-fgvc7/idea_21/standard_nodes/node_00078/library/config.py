import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the SAM-Optimized Stratified Shuffle-Split Ensemble pipeline.
    Defines hyperparameters, paths, and structural settings for Idea 21.
    """

    # ==========================================
    # Experiment Metadata
    # ==========================================
    IDEA_NAME = "idea_21"
    DEBUG = False

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Metadata (Pre-generated)
    TRAIN_METADATA_PATH = "./metadata/train_metadata.csv"
    VAL_METADATA_PATH = "./metadata/val_metadata.csv"
    TEST_METADATA_PATH = "./metadata/test_metadata.csv"

    # Working Directories
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    OUTPUT_DIR = os.path.join(WORKING_DIR, "output")
    MODEL_DIR = os.path.join(WORKING_DIR, "models")
    CACHE_DIR = WORKING_DIR

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    IMAGE_SIZE = 256
    NUM_CLASSES = 4
    CLASS_NAMES = ["healthy", "multiple_diseases", "rust", "scab"]

    # Augmentation Constraints
    # Strictly limit ShiftScaleRotate scaling to +/- 5% to preserve small lesions
    SHIFT_SCALE_ROTATE_LIMIT = 0.05

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    MODEL_NAME = "resnet34"
    PRETRAINED = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Optimizer: Sharpness-Aware Minimization (SAM)
    USE_SAM = True
    SAM_RHO = 0.05

    # Scheduler: Cosine Annealing Warm Restarts
    # Cycle length strictly synchronized to total epochs
    T_0 = 15
    MIN_LR = 1e-6

    # ==========================================
    # Ensemble / Topology Strategy
    # ==========================================
    # Stratified Shuffle-Split Ensemble
    # We use 5 unique random splits instead of fixed folds
    N_SPLITS = 5

    # Widely spaced seeds for the 5 unique splits to minimize RNG correlation
    SEEDS = [2021, 2022, 2023, 2024, 2025]

    # ==========================================
    # Hardware & Reproducibility
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # ==========================================
    # Inference
    # ==========================================
    USE_TTA = False  # Explicitly disable Test-Time Augmentation

    @classmethod
    def setup(cls, debug=False):
        """
        Initialize directories and adjust configuration for debugging.

        Args:
            debug (bool): If True, reduces epochs and splits for rapid testing.
        """
        cls.DEBUG = debug

        if cls.DEBUG:
            print("DEBUG MODE: Enabled")
            cls.EPOCHS = 2
            cls.N_SPLITS = 2
            cls.SEEDS = cls.SEEDS[:2]

        # Create necessary directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        print(f"Directories ready at {cls.WORKING_DIR}")
        print(
            f"Configuration: {cls.N_SPLITS} splits, {cls.EPOCHS} epochs, Device: {cls.DEVICE}"
        )


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across all libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
