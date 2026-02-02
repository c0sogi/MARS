import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # ==========================================
    # Project Structure & Paths
    # ==========================================
    PROJECT_NAME = "idea_24"

    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directories (Write Access)
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMAGE_SIZE = 32
    NUM_CHANNELS = 3
    NUM_CLASSES = 1  # Binary classification

    # Caching
    # If True, tries to load pre-processed tensors from disk
    USE_CACHE = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    N_FOLDS = 5

    # Batch Size
    # 32x32 images are very small, allowing for larger batch sizes.
    BATCH_SIZE = 128

    # Learning Rate Strategy (SWA)
    # Phase 1: Convergence
    EPOCHS_CONVERGENCE = 25
    LR_MAX = 1e-3
    LR_MIN = 1e-5

    # Phase 2: SWA Exploration
    EPOCHS_SWA = 10
    SWA_LR = 1e-3  # High constant LR for exploration

    TOTAL_EPOCHS = EPOCHS_CONVERGENCE + EPOCHS_SWA

    # Optimization
    WEIGHT_DECAY = 1e-4
    OPTIMIZER_NAME = "AdamW"

    # Regularization
    MIXUP_ALPHA = 0.2  # Mild mixup

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "MultiHeadRepVGG"
    MODEL_WIDTH_MULTIPLIER = 1.0
    USE_CONSERVATIVE_STEM = True  # Preserve 32x32 resolution initially

    # ==========================================
    # Compute & Debugging
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging / Quick Run
    # Set DEBUG to True to run on a small subset of data
    DEBUG = False
    DEBUG_SUBSET_SIZE = 1000

    @classmethod
    def setup(cls):
        """
        Creates the necessary working directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seed immediately upon setup
        seed_everything(cls.SEED)

        print(f"Configuration setup complete. Working directory: {cls.WORKING_DIR}")
        print(f"Device: {cls.DEVICE}")
