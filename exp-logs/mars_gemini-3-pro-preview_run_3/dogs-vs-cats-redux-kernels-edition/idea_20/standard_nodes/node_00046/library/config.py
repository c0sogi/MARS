import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration module for the 5-Fold Stratified Heterogeneous Ensemble.
    Defines hyperparameters, file paths, and model-specific configurations
    for the Dog vs Cat classification task.
    """

    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100  # Number of samples per fold to use in debug mode

    # Compute
    NUM_WORKERS = 4  # optimized for the available vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    # The training strategy involves 5-fold CV on the full dataset.
    # The training script should combine train and val metadata if necessary.
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working and Output Directories
    WORKING_DIR = "./working/idea_20"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Training Strategy
    # -------------------------------------------------------------------------
    NUM_FOLDS = 5

    # Global Optimization Settings
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    MIN_LR = 1e-6

    # Ensemble Safety
    # OOF Log Loss threshold to filter out "Poison Pill" models before ensembling
    OOF_THRESHOLD = 0.5

    # -------------------------------------------------------------------------
    # Model Architectures (The Golden Trio)
    # -------------------------------------------------------------------------
    # Maps architecture names to their specific training requirements.
    # Resolutions and Epochs are decoupled to maximize inductive bias diversity.
    MODEL_CONFIGS = {
        "resnet50": {
            "model_name": "resnet50.a1_in1k",
            "img_size": 256,
            "epochs": 8,
            "batch_size": 64,
            "resize_scale": (0.8, 1.0),
        },
        "convnext_small": {
            "model_name": "convnext_small.fb_in1k",
            "img_size": 288,
            "epochs": 8,
            "batch_size": 32,
            "resize_scale": (0.8, 1.0),
        },
        "maxvit_tiny": {
            "model_name": "maxvit_tiny_tf_224.in1k",
            "img_size": 224,
            "epochs": 15,
            "batch_size": 32,
            "resize_scale": (0.8, 1.0),
        },
    }

    # -------------------------------------------------------------------------
    # Setup Utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def set_seed(seed=42):
        """Sets fixed random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    @classmethod
    def setup(cls):
        """Initializes the environment, directories, and seeds."""
        # Create necessary directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Apply seeding
        cls.set_seed(cls.SEED)


# Initialize environment immediately upon import
Config.setup()
