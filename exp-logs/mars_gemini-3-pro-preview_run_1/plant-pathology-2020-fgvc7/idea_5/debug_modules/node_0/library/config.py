import os
import torch


class Config:
    """
    Global configuration for the Apple Disease Detection task.
    Implements the Dual-Architecture Heterogeneous Ensemble strategy (Idea 5).
    """

    # ==========================================
    # General Setup
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SUBSET_SIZE = 50

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Metadata files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output directories
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 256
    NUM_CLASSES = 4
    CLASS_LABELS = ["healthy", "multiple_diseases", "rust", "scab"]

    # ==========================================
    # Model Configuration
    # ==========================================
    # Heterogeneous Ensemble: ResNet34 (Anchor) + ResNeXt50 (High Capacity)
    MODEL_ARCHS = ["resnet34", "resnext50_32x4d"]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    N_FOLDS = 5
    EPOCHS = 15
    BATCH_SIZE = 32

    # Optimizer settings
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4

    # Scheduler settings (Cosine Annealing Warm Restarts)
    # T_0 synchronized with EPOCHS to ensure full decay within budget
    T_0 = EPOCHS
    T_MULT = 1
    ETA_MIN = 1e-6

    # ==========================================
    # System & Compute
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def create_dirs(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
