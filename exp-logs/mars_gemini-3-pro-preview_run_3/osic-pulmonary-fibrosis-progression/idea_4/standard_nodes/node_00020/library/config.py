import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the Dual-Path Transformer-Fused Network experiment.
    """

    # ==========================================
    # Experiment Identification
    # ==========================================
    EXPERIMENT_NAME = "idea_4"
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging
    DEBUG_SIZE = 50  # Number of samples to use if DEBUG is True

    # ==========================================
    # Data Paths
    # ==========================================
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # DICOM directories
    TRAIN_DICOM_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DICOM_DIR = os.path.join(INPUT_DIR, "test")

    # Working directories (Write Allowed)
    WORKING_DIR = os.path.join("./working", EXPERIMENT_NAME)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Hyperparameters
    # ==========================================
    # Data / Preprocessing
    N_SLICES = 3  # Mandatory: Apical, Middle, Basal
    IMG_SIZE = 256  # Input size for EfficientNet

    # Normalization Statistics (from EDA)
    # Target FVC Standardization: (FVC - Mean) / Std
    TARGET_MEAN = 2654.6528
    TARGET_STD = 801.7017

    # Training
    EPOCHS = 35  # Mandatory: Extended duration for convergence
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    NUM_WORKERS = 2

    # Model Architecture
    # Path 1 (Transformer)
    EMBED_DIM = 128  # Dimension D for projection
    TRANSFORMER_HEADS = 4
    TRANSFORMER_LAYERS = 2
    BACKBONE_NAME = "efficientnet_b0"

    # Path 2 (Linear) & Fusion
    # No specific params needed here, defined in model logic

    # Metric
    SIGMA_CLIP = 70.0
    MAX_ERROR = 1000.0

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=42):
    """
    Sets fixed random seeds for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


def setup_directories():
    """
    Creates necessary working directories for cache, checkpoints, and submissions.
    """
    dirs = [
        Config.WORKING_DIR,
        Config.CACHE_DIR,
        Config.CHECKPOINT_DIR,
        Config.SUBMISSION_DIR,
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print(f"Directories initialized at {Config.WORKING_DIR}")


# Initialize setup immediately when module is imported to ensure environment is ready
seed_everything(Config.SEED)
setup_directories()
