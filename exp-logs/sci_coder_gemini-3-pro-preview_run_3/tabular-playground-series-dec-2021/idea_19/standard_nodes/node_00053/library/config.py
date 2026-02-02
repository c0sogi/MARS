import os
import random
import numpy as np
import torch


class Config:
    # Experiment Reproducibility
    SEED = 42

    # Data Paths
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working Directories
    WORKING_DIR = "./working/idea_19"
    CACHE_DIR = WORKING_DIR  # Directory for caching processed data
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Hyperparameters
    BATCH_SIZE = 4096
    EPOCHS = 60
    LEARNING_RATE = 1e-3

    # Architecture Specifics
    HIDDEN_DIM = 512
    LOW_RANK_FACTOR = 16
    DROPOUT = 0.1

    # Data Specifics
    # Cover_Type classes are integers 1-7 (though class 5 might be missing)
    # We will assume 7 output classes (mapping 1->0, ..., 7->6)
    NUM_CLASSES = 7

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Using 12 vCPUs as provided in the environment
    NUM_WORKERS = 12


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_directories():
    """
    Ensures that the working and submission directories exist.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
setup_directories()
