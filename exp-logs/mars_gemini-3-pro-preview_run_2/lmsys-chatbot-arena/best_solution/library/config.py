import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration class for the Siamese DeBERTa-Base project.
    """

    # Model Architecture
    MODEL_NAME = "microsoft/deberta-v3-xsmall"
    MAX_LENGTH = 512

    # Training Hyperparameters
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    EPOCHS = 2
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 10.0
    SCHEDULER_TYPE = "cosine"
    NUM_WARMUP_STEPS_RATIO = 0.1

    # Compute
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    SEED = 42

    # Paths
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories and Files
    WORKING_DIR = "./working/idea_3"
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

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

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
