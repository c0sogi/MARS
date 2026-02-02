import os
import torch
import random
import numpy as np


class Config:
    """
    Centralized configuration for the pathology tumor detection task.
    """

    # --- Project & Paths ---
    PROJECT_NAME = "idea_1"
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Derived Paths
    ARTIFACT_DIR = os.path.join(WORKING_DIR, PROJECT_NAME)
    CHECKPOINT_DIR = os.path.join(ARTIFACT_DIR, "checkpoints")
    CACHE_DIR = os.path.join(ARTIFACT_DIR, "cache")
    PREDICTION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # --- Data Parameters ---
    ORIGINAL_IMG_SIZE = 96
    CROP_SIZE = 64  # Input size for the model (context around 32x32 center)
    ROI_SIZE = 32  # The region that determines the label

    # Normalization (ImageNet defaults)
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # --- Model Parameters ---
    MODEL_NAME = "resnet18"  # Cite {solution_lesson_node_00002}
    PRETRAINED = True
    NUM_CLASSES = 1  # Binary classification
    DROPOUT_RATE = 0.2

    # --- Training Hyperparameters ---
    SEED = 42
    BATCH_SIZE = 256  # A100 40GB allows for larger batch sizes
    NUM_WORKERS = 4  # Number of dataloader workers

    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    EPOCHS = 10  # Cite {solution_lesson_node_00002}
    EARLY_STOPPING_PATIENCE = 3

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 1000  # Number of samples to use if DEBUG is True

    # --- Compute ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Ensures all necessary working directories exist.
        """
        os.makedirs(cls.ARTIFACT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


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
