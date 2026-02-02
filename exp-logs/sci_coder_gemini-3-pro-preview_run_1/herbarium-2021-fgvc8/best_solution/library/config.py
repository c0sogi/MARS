import os
import torch
import random
import numpy as np


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    """
    Central configuration for the Plant Species Classification task.
    Handles paths, hyperparameters, and compute settings.
    """

    # ==========================================
    # Project Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    IDEA_DIR = "idea_3"

    # Cache directory for intermediate files (e.g., model checkpoints, processed maps)
    CACHE_DIR = os.path.join(WORKING_DIR, IDEA_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    TRAIN_METADATA_JSON = os.path.join(INPUT_DIR, "train", "metadata.json")

    # Output Paths
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    MODEL_PATH = os.path.join(CACHE_DIR, "model.pth")
    FAMILY_MAPPING_CACHE = os.path.join(CACHE_DIR, "species_family_map.parquet")

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    IMG_SIZE = 224
    NUM_CLASSES = 64500

    # Hard Undersampling: Maximum number of images per species to keep in training
    MAX_SAMPLES_PER_CLASS = 50

    # Data Loading
    NUM_WORKERS = 12
    PIN_MEMORY = True

    # ==========================================
    # Model & Training Hyperparameters
    # ==========================================
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True

    # Multi-Task Learning: Weight for the auxiliary Family classification loss
    FAMILY_LOSS_WEIGHT = 0.5

    # Optimization
    BATCH_SIZE = 128
    NUM_EPOCHS = 10
    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    # Compute
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Debugging & Development
    # ==========================================
    # If DEBUG is True, the dataset will be truncated to DEBUG_SAMPLES
    DEBUG = False
    DEBUG_SAMPLES = 5000
