import os
import torch
import random
import numpy as np


class Config:
    # ==========================
    # System & Reproducibility
    # ==========================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers

    # ==========================
    # File Paths
    # ==========================
    # Root directories
    INPUT_ROOT = "./input"
    METADATA_ROOT = "./metadata"
    WORKING_ROOT = "./working"

    # Specific Idea Directory for Caching and Outputs
    IDEA_ID = "idea_29"
    OUTPUT_DIR = os.path.join(WORKING_ROOT, IDEA_ID)
    CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Input Metadata Files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_ROOT, "train.csv")
    VAL_CSV = os.path.join(METADATA_ROOT, "val.csv")
    TEST_CSV = os.path.join(METADATA_ROOT, "test.csv")

    # DICOM Directories
    TRAIN_DICOM_DIR = os.path.join(INPUT_ROOT, "train")
    TEST_DICOM_DIR = os.path.join(INPUT_ROOT, "test")

    # ==========================
    # Data Preprocessing
    # ==========================
    IMG_SIZE = 224
    SLAB_COUNT = 3  # Tri-Slab
    OVERLAP = 0.15  # 15% overlap

    # Tabular Features
    # Note: 'Weeks' is used for trajectory calculation, not input to the static embedding MLP
    TABULAR_COLS = ["Age", "Percent", "Sex", "SmokingStatus"]

    # ==========================
    # Model Architecture
    # ==========================
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True
    HIDDEN_DIM = 1280  # Native dimensionality of EfficientNet-B0 GAP

    # Projection settings
    TABULAR_HIDDEN_DIM = 512
    FUSION_HEADS = 4
    FUSION_LAYERS = 1

    # Output: alpha (slope), sigma_base, sigma_growth
    OUTPUT_DIM = 3

    # ==========================
    # Training Hyperparameters
    # ==========================
    N_EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1.0

    # Scheduler (Cosine Annealing)
    T_MAX = N_EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 8

    # Metric Constraints
    MAX_ERROR = 1000
    MIN_CONFIDENCE = 70

    @staticmethod
    def setup_directories():
        """Creates necessary directories for caching and checkpoints."""
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        # Ensure submission directory exists
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    @staticmethod
    def set_seed(seed=42):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Initialize environment immediately upon import
Config.setup_directories()
Config.set_seed(Config.SEED)
