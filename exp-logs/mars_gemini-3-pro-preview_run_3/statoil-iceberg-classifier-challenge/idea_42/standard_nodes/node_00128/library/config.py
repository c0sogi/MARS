import os
import torch
import random
import numpy as np


class Config:
    # Project Meta
    PROJECT_NAME = "Iceberg_Classifier_AAHA_CNN"
    SEED = 42
    DEBUG = False  # Set to True to use a subset of data for debugging

    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_42"

    # Sub-directories (will be created by setup)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Data Specifications
    IMAGE_HEIGHT = 75
    IMAGE_WIDTH = 75
    IN_CHANNELS = 3  # HH, HV, Average((HH+HV)/2)

    # Model Architecture Settings
    MODEL_NAME = "AAHA_CNN"
    DROPOUT_RATE = 0.5
    BLUR_KERNEL_SIZE = 3  # For MaxBlurPool

    # Training Hyperparameters
    N_FOLDS = 5
    NUM_EPOCHS = 75
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3  # Constant learning rate as per idea
    PATIENCE = 12  # Early stopping patience
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_debug_config(cls):
        """Returns a modified configuration for debugging purposes."""
        cls.DEBUG = True
        cls.NUM_EPOCHS = 2
        cls.N_FOLDS = 2
        return cls


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
