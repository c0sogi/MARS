import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration class for the FTU Detection Pipeline.
    Centralizes all hyperparameters, file paths, and setup configurations.
    """

    # ==============================
    # File Paths & Directories
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary write directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==============================
    # Data Configuration
    # ==============================
    TILE_SIZE = 1024
    # Ratio of purely background tiles to keep during training (undersampling)
    BACKGROUND_SAMPLE_RATE = 0.20
    NUM_WORKERS = 8  # Utilizing available vCPUs

    # ==============================
    # Model Configuration
    # ==============================
    ENCODER = "resnet18"
    ENCODER_WEIGHTS = "imagenet"
    CLASSES = 1

    # ==============================
    # Training Configuration
    # ==============================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Batch size optimized for available GPU memory (A100 40GB)
    BATCH_SIZE = 16
    NUM_EPOCHS = 20

    # Optimizer settings (AdamW)
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Scheduler settings (CosineAnnealingLR)
    T_MAX = NUM_EPOCHS
    ETA_MIN = 1e-6

    # Debugging / Development flags
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use if DEBUG is True

    # ==============================
    # Inference Configuration
    # ==============================
    # Threshold for converting probability maps to binary masks
    THRESHOLD = 0.5
    # Overlap for sliding window inference (0.5 = 50% overlap)
    INFERENCE_OVERLAP = 0.5

    @staticmethod
    def seed_everything(seed: int = 42):
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


# Apply the seed immediately upon import to ensure reproducibility
Config.seed_everything(Config.SEED)
