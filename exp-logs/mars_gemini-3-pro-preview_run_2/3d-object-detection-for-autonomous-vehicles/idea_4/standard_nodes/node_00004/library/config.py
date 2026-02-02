import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # 1. Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Calibration Files (referenced in metadata, but good to have base path)
    TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "train_data")
    TEST_DATA_DIR = os.path.join(INPUT_DIR, "test_data")

    # ==========================================
    # 2. Data Parameters
    # ==========================================
    # Input Resolution (Width, Height) - Stride 32 compatibility preferred
    # 800x448 is a reasonable compromise for speed/accuracy on A100
    INPUT_WIDTH = 800
    INPUT_HEIGHT = 448

    # CenterNet Stride (Output resolution will be Input / 4)
    DOWN_RATIO = 4
    OUTPUT_WIDTH = INPUT_WIDTH // DOWN_RATIO
    OUTPUT_HEIGHT = INPUT_HEIGHT // DOWN_RATIO

    # Classes identified in EDA
    CLASS_NAMES = [
        "car",
        "other_vehicle",
        "pedestrian",
        "bicycle",
        "truck",
        "bus",
        "motorcycle",
        "animal",
        "emergency_vehicle",
    ]
    NUM_CLASSES = len(CLASS_NAMES)
    CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}

    # Image Normalization (ImageNet defaults)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Max objects to process per image during training
    MAX_OBJS = 100

    # ==========================================
    # 3. Model Parameters
    # ==========================================
    BACKBONE = "resnet34"
    HEAD_CONV = 64  # Channels for intermediate head layers

    # ==========================================
    # 4. Training Parameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1.25e-4
    NUM_EPOCHS = 15
    NUM_WORKERS = 4

    # Debug / Subset options
    DEBUG = False
    DATA_SUBSET_RATIO = 0.1  # Used if DEBUG is True

    # ==========================================
    # 5. Inference Parameters
    # ==========================================
    CONF_THRESHOLD = 0.3
    TOP_K = 50

    # ==========================================
    # 6. Hardware
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    @staticmethod
    def setup():
        """Creates necessary working directories."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    @staticmethod
    def set_seed(seed=None):
        """Sets fixed random seeds for reproducibility."""
        if seed is None:
            seed = Config.SEED

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
