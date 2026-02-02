import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    """
    Configuration class to centralize all hyperparameters, file paths, and settings
    for the Cassava Leaf Disease Classification task.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata paths (pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output paths
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Artifacts
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "resnet18_baseline.pth")

    # ==========================================
    # Model & Data Hyperparameters
    # ==========================================
    MODEL_NAME = "resnet50"
    NUM_CLASSES = 5
    IMG_SIZE = 224
    CHANNELS = 3

    # Normalization Statistics (ImageNet)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Training Settings
    SEED = 42
    # Cite solution_lesson_node_00008: Implicit Regularization via Batch Size Reduction
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Compute Settings
    NUM_WORKERS = 8  # Optimized for 12 vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Debugging / Development
    # ==========================================
    # If True, limits the dataset size for rapid testing of the pipeline
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    def __init__(self):
        """
        Initializes the configuration and ensures necessary output directories exist.
        """
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)
