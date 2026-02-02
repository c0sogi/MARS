import os
import torch
import numpy as np
import random


class Config:
    """
    Global configuration and hyperparameters for the Apple Disease Detection task.
    Centralizes all settings for data loading, model training, and inference.
    """

    # --- Reproducibility ---
    SEED = 42

    # --- Paths ---
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata directories (Generated in previous steps)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output directories (Writeable)
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Configuration ---
    IMG_SIZE = 256
    NUM_CLASSES = 6
    # Class labels derived from EDA (order matters for consistency)
    CLASSES = [
        "scab",
        "healthy",
        "frog_eye_leaf_spot",
        "rust",
        "complex",
        "powdery_mildew",
    ]

    # --- Debugging / Development ---
    # Set to True to train on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    # --- Model Configuration ---
    # Using EfficientNetV2-Small (Fused-MBConv architecture) via timm
    MODEL_NAME = "tf_efficientnetv2_s.in1k"

    # --- Training Configuration ---
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    # Adjust based on available vCPUs (12 available)
    NUM_WORKERS = 4

    # --- Compute ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def setup_reproducibility(seed=42):
        """
        Sets random seeds for Python, NumPy, and PyTorch to ensure reproducible results.

        Args:
            seed (int): The seed value to use.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Enforce deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        os.environ["PYTHONHASHSEED"] = str(seed)


# Initialize necessary writeable directories
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
