import os
import torch
import random
import numpy as np


class Config:
    """
    Centralized configuration for the Breast Cancer Detection task.
    Implements settings for the Baseline Approach: Naive MIL with ResNet-18.
    """

    # --------------------------------------------------------------------------
    # Directory and File Paths
    # --------------------------------------------------------------------------
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata paths (pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working directory for caching processed data and model checkpoints
    # Specifically for idea_1 (ResNet18 Baseline)
    WORKING_DIR = "./working/idea_1"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Model checkpoint location
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "resnet18_baseline.pth")

    # Submission output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing Parameters
    # --------------------------------------------------------------------------
    # Image dimensions: (Height, Width)
    # 512x512 is the chosen sweet spot for the baseline
    IMG_SIZE = (512, 512)

    # Number of channels (Mammograms are grayscale, but ResNet expects 3 usually.
    # We will replicate channels or modify the first layer in the model definition)
    IN_CHANNELS = 3

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Balanced Sampling: Target ratio of positive samples in a batch
    # 0.5 ensures equal representation of cancer/non-cancer in every batch
    POSITIVE_SAMPLING_RATIO = 0.5

    # Target prior probability (natural prevalence) for recalibration
    # Based on metadata analysis (~2%)
    TARGET_PRIOR = 0.02

    # Early Stopping settings
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_MIN_DELTA = 0.0001

    # --------------------------------------------------------------------------
    # Hardware & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42

    # Use CUDA if available
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Number of data loading workers
    # 12 vCPUs available -> using 8 is a safe high-performance choice
    NUM_WORKERS = 8

    @staticmethod
    def set_seed(seed=42):
        """
        Sets fixed random seeds for Python, NumPy, and PyTorch to ensure
        reproducible results.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set environment variable for hash seeding
        os.environ["PYTHONHASHSEED"] = str(seed)


# Apply seeding immediately upon import
Config.set_seed(Config.SEED)
