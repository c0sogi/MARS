import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # --- Experiment Configuration ---
    EXPERIMENT_NAME = "idea_6"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 50

    # --- Hardware ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Optimized for the 12 vCPU environment

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", EXPERIMENT_NAME)

    # Input Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Directories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Output Files
    MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data parameters ---
    IMG_SIZE = 224  # EfficientNet-B0 standard input size
    NUM_SLICES = 3  # CAP-Net uses 3 slices (Apical, Middle, Basal)

    # --- Feature Definitions ---
    ID_COL = "Patient"
    TARGET_COL = "FVC"

    # Numerical features to be standardized and fed into MLP
    NUMERICAL_COLS = ["Weeks", "Percent", "Age"]

    # Categorical features to be embedded
    CATEGORICAL_COLS = ["Sex", "SmokingStatus"]

    # --- Model Hyperparameters ---
    BACKBONE = "efficientnet_b0"
    PROJECTION_DIM = 256  # Increased capacity for fine-tuning
    HIDDEN_DIM = 512  # Increased capacity for Fusion MLP

    # --- Training Hyperparameters ---
    EPOCHS = 50  # Extended duration for fine-tuning
    BATCH_SIZE = 32  # Fits within A100 40GB VRAM
    HEAD_LR = 1e-3
    BACKBONE_LR = 1e-4
    WEIGHT_DECAY = 1e-2
    PATIENCE = 10  # Early stopping patience

    # --- Metric Constants ---
    METRIC_CLIP_SIGMA = 70
    METRIC_MAX_ERR = 1000

    @classmethod
    def setup(cls):
        """
        Initializes the experiment environment:
        1. Sets random seeds.
        2. Creates necessary directories for caching, checkpoints, and submission.
        """
        seed_everything(cls.SEED)

        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Ensure working directory exists (redundant but safe)
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
