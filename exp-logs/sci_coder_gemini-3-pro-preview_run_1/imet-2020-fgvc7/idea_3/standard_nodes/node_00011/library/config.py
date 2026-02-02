import os
import torch
import random
import numpy as np


class Config:
    # Global Seeds
    SEED = 42

    # Directory Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Image Directories (Relative to INPUT_DIR based on metadata logic)
    # Note: Metadata contains paths like "train/id.png", so we join with INPUT_DIR

    # Model Hyperparameters
    MODEL_NAME = "convnext_small.fb_in22k"  # Using ImageNet-22k pretrained weights for better feature extraction
    IMG_SIZE = 320
    NUM_CLASSES = 3474
    IN_CHANNELS = 3

    # Training Hyperparameters
    EPOCHS = 18
    BATCH_SIZE = 64  # Adjusted for A100 40GB with 320px resolution and Small model
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 10.0

    # Loss Configuration
    POS_WEIGHT = 12.0  # To handle class sparsity
    LABEL_SMOOTHING = 0.05

    # Optimization & Scheduling
    T_MAX = EPOCHS  # For Cosine Annealing
    MIN_LR = 1e-6

    # Hardware & Performance
    NUM_WORKERS = 4
    PIN_MEMORY = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Advanced Strategies
    USE_EMA = True
    EMA_DECAY = 0.9999
    USE_AMP = True  # Automatic Mixed Precision

    # Inference
    THRESHOLD_SEARCH_START = 0.01
    THRESHOLD_SEARCH_END = 0.99
    THRESHOLD_SEARCH_STEP = 0.01

    @classmethod
    def setup(cls):
        """Ensures working directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)


def seed_everything(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Automatically setup directories when module is imported or Config is used
Config.setup()
