import os
import torch
import random
import numpy as np


class Config:
    # ==== General Settings ====
    SEED = 42
    DEBUG = False  # Set to True to use a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    # ==== File Paths ====
    # Input Metadata (Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output / Working Directory
    # We use idea_4 as the specific directory for this iteration
    WORKING_DIR = "./working/idea_4"
    CACHE_DIR = WORKING_DIR  # Directory for caching processed datasets

    # Model Checkpoint
    MODEL_FILENAME = "best_model.pth"
    MODEL_PATH = os.path.join(WORKING_DIR, MODEL_FILENAME)

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==== Model Hyperparameters ====
    MODEL_NAME = "microsoft/deberta-v3-small"
    MAX_LENGTH = 512
    NUM_LABELS = 3  # Winner A, Winner B, Tie

    # ==== Training Hyperparameters ====
    TRAIN_BATCH_SIZE = 16  # A100 40GB can handle larger, but 16 is safe for 512 seq len
    VALID_BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    EPOCHS = 3
    WARMUP_RATIO = 0.1
    GRADIENT_CLIPPING = 1.0

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 2

    # ==== Hardware ====
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
