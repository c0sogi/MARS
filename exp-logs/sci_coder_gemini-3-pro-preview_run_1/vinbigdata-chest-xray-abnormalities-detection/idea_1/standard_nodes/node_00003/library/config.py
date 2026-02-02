import os
import torch
import random
import numpy as np


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")

    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Configuration
    # =========================================================================
    IMG_SIZE = 512
    NUM_CLASSES = 14  # Classes 0-13. Class 14 is "No finding" (handled via logic)

    # Image Normalization (ImageNet defaults)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # =========================================================================
    # Model Configuration (CenterNet-like)
    # =========================================================================
    BACKBONE = "resnet18"
    DOWN_RATIO = 4  # Stride of the output feature map (512 / 4 = 128)
    MAX_OBJECTS = 128  # Maximum number of objects per image for tensor construction

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 8  # Optimized for 12 vCPUs

    # =========================================================================
    # Inference Configuration
    # =========================================================================
    CONF_THRESHOLD = 0.2
    IOU_THRESHOLD = (
        0.4  # Not strictly used for CenterNet decoding but useful for validation
    )
    TOP_K = 50  # Max detections to return per image

    # =========================================================================
    # Debugging & Development
    # =========================================================================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200  # Number of images to use when DEBUG is True

    # =========================================================================
    # Compute
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
