import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the 'Calibrated Full-Data Seed Ensemble with Discriminative Fine-Tuning' strategy.
    Centralizes all hyperparameters, paths, and settings.
    """

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching and outputs (Idea 13 specific)
    WORKING_DIR = "./working/idea_13"
    OUTPUT_DIR = os.path.join(WORKING_DIR, "output")
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Paths
    # The strategy requires combining train and val for Phase 2 (100% data)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Image Source Directory
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Final Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    # Resolution set to 256x256 to ensure pipeline consistency and avoid batch instability
    IMG_SIZE = (256, 256)

    # Batch Size
    BATCH_SIZE = 32

    # Data Loading Workers
    NUM_WORKERS = 4

    # Target Labels
    TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]
    NUM_CLASSES = len(TARGET_COLS)

    # ==========================================
    # Model Configuration
    # ==========================================
    MODEL_NAME = "resnet34"
    PRETRAINED = True

    # Initial Loss Check Threshold (-ln(1/4) approx 1.38)
    # Used to verify correct weight initialization before training
    INITIAL_LOSS_THRESHOLD = 1.38

    # ==========================================
    # Training Configuration
    # ==========================================
    # Discriminative Fine-Tuning Learning Rates
    # Uniform LR preferred for small datasets to prevent overfitting (Cite solution_lesson_node_00044)
    BACKBONE_LR = 1e-4
    HEAD_LR = 1e-4

    # Optimizer settings
    WEIGHT_DECAY = 1e-4

    # Phase 1: Calibration (Finding Optimal Epoch)
    # Stratified 5-Fold CV to determine E_opt
    N_FOLDS = 5
    MAX_EPOCHS = 20

    # Phase 2: Seed Ensemble (Production)
    # 5 distinct seeds for variance reduction on full data
    SEEDS = [42, 2024, 1337, 7, 99]

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Utility Functions
    # ==========================================
    @staticmethod
    def set_seed(seed: int = 42):
        """
        Sets fixed random seeds for reproducibility across random, numpy, and torch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
