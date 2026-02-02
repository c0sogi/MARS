import os
import torch
import random
import numpy as np


class Config:
    """
    Centralized configuration for the 2.5D ShuffleNet-PSPNet segmentation task.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Metadata Files (Generated in previous steps)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Outputs
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    IMG_SIZE = (256, 256)
    IN_CHANNELS = 3  # 2.5D Input: Slice i-1, i, i+1

    # Class Definitions
    CLASS_LABELS = ["large_bowel", "small_bowel", "stomach"]
    NUM_CLASSES = len(CLASS_LABELS)
    CLASS_ID_MAP = {label: idx for idx, label in enumerate(CLASS_LABELS)}

    # Physical Properties
    SLICE_DEPTH_MM = 3.0

    # Sampling Strategy
    # Ratio of negative samples (no mask) to keep during training to handle imbalance
    NEGATIVE_SAMPLE_RATIO = 0.5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    EPOCHS = 15

    # Optimization (SGD + Poly Scheduler)
    LEARNING_RATE = 1e-2
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    POLY_POWER = 0.9

    # Loss Function Weights (Composite Loss)
    LOSS_CE_WEIGHT = 0.5
    LOSS_DICE_WEIGHT = 0.5

    # =========================================================================
    # Inference & Post-processing
    # =========================================================================
    THRESHOLD = 0.5
    MIN_PIXELS = 1  # Minimum pixels to consider a mask valid

    # =========================================================================
    # System & Debugging
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing available vCPUs
    SEED = 42

    # Debug configuration
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200  # Number of samples to use when DEBUG is True

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @staticmethod
    def seed_everything(seed=42):
        """
        Sets seeds for reproducibility across random, numpy, and torch.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
