import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    # Ensure submission directory exists
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data / Preprocessing
    # ==========================================
    IMAGE_SIZE = 512  # Fixed resolution as per Idea
    NUM_CLASSES = 11

    # Target Columns in order
    TARGET_COLS = [
        "ETT - Abnormal",
        "ETT - Borderline",
        "ETT - Normal",
        "NGT - Abnormal",
        "NGT - Borderline",
        "NGT - Incompletely Imaged",
        "NGT - Normal",
        "CVC - Abnormal",
        "CVC - Borderline",
        "CVC - Normal",
        "Swan Ganz Catheter Present",
    ]

    # ==========================================
    # Model
    # ==========================================
    MODEL_NAME = "mobilenet_v3_large"
    PRETRAINED = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4  # 12 vCPUs available

    # Debugging / Development
    DEBUG = False
    MAX_TRAIN_SAMPLES = None  # Set to an integer to limit data for quick debugging
    MAX_VAL_SAMPLES = None

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


def seed_everything(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Initialize directories
Config.setup()
