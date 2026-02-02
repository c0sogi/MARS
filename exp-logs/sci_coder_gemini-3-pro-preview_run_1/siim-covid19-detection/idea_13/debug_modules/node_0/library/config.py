import os
import torch
import numpy as np
import random


def seed_everything(seed=42):
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


class Config:
    """
    Global configuration for the ResNet18-D U-Net experiment.
    """

    # General
    SEED = 42
    DEBUG = False  # Toggle for debugging with smaller dataset

    # Compute
    NUM_WORKERS = 12  # Utilizing available vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"

    # Ensure working directory exists for caching/checkpoints
    os.makedirs(WORKING_DIR, exist_ok=True)

    # File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Parameters
    IMG_SIZE = 512
    NUM_CLASSES = 4

    # Training Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1e-4  # Conservative base learning rate
    WEIGHT_DECAY = 1e-2

    # Loss Weights: [Classification, Segmentation]
    # 1:10 ratio prioritizes dense prediction
    LOSS_WEIGHTS = [1.0, 10.0]

    # Model Architecture
    BACKBONE = "resnet18d"  # ResNet18-D (Deep Stem)


# Apply seeding immediately
seed_everything(Config.SEED)
