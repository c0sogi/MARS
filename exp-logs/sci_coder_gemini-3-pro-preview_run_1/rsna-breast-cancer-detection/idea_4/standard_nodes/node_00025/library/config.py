import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for Breast Cancer Detection pipeline.
    Defines hyperparameters, file paths, and model settings.
    """

    # ==========================================
    # General Configuration
    # ==========================================
    PROJECT_NAME = "BreastCancerDetection_Idea4"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    # ==========================================
    # Directory Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    # Note: Usually done in main, but paths are defined here.
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_4")

    # Metadata Files (Pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # ==========================================
    # Data Configuration
    # ==========================================
    # Input strategy: Spatial Channel Expansion (Image + Age + Implant)
    # Dimensions: [Batch, 3, 512, 512]
    IMAGE_SIZE = (512, 512)
    INPUT_CHANNELS = 3
    NUM_CLASSES = 1

    # ==========================================
    # Model Configuration
    # ==========================================
    MODEL_NAME = "efficientnet_b2"
    # Probability that Age/Implant channels are zeroed out during training
    MODALITY_DROPOUT_PROB = 0.5

    # ==========================================
    # Training Configuration
    # ==========================================
    BATCH_SIZE = 24  # Adjusted for A100 40GB and 768x768 resolution
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Loss Function Weights
    # Inverse class frequency weighting (~47.0 based on data analysis)
    POS_WEIGHT = 47.0

    # Optimization
    USE_GRAD_CLIPPING = False  # Explicitly disabled as per strategy

    # ==========================================
    # Hardware Configuration
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing available vCPUs
    PIN_MEMORY = True

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("\n" + "=" * 40)
        print(f"CONFIGURATION: {cls.PROJECT_NAME}")
        print("=" * 40)
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key.ljust(25)}: {value}")
        print("=" * 40 + "\n")


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to: {seed}")
