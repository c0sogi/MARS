import os
import torch
import numpy as np
import random


class Config:
    # =========================================================================
    # Directories & File Paths
    # =========================================================================
    PROJECT_NAME = "idea_6"
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    LABELS_PATH = os.path.join(INPUT_DIR, "labels.csv")

    # Output Files
    MODEL_PATH = os.path.join(WORKING_DIR, "convnext_small_best.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    IMG_SIZE = 320
    NUM_CLASSES = 3474
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # =========================================================================
    # Model Configuration
    # =========================================================================
    MODEL_NAME = "convnext_small"  # timm backbone
    PRETRAINED = True
    USE_GEM = True  # Generalized Mean Pooling
    USE_EMA = True  # Model Exponential Moving Average
    EMA_DECAY = 0.9999

    # =========================================================================
    # Training Configuration
    # =========================================================================
    SEED = 42
    EPOCHS = 20
    BATCH_SIZE = 64  # Fits A100-40GB with 320x320 resolution

    # Optimizer (AdamW)
    LR = 2e-4
    WEIGHT_DECAY = 0.05

    # Scheduler (Cosine Annealing)
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 1

    # =========================================================================
    # Loss & Regularization
    # =========================================================================
    # BCEWithLogitsLoss
    POS_WEIGHT = 12.0  # Weight for positive class to improve recall

    # Label Smoothing
    LABEL_SMOOTHING = 0.05

    # Mixup & CutMix
    USE_MIXUP = True
    MIXUP_ALPHA = 0.8
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 0.5  # Probability of applying mixup/cutmix

    # =========================================================================
    # Inference Configuration
    # =========================================================================
    THRESHOLD_START = 0.01
    THRESHOLD_END = 0.99
    THRESHOLD_STEP = 0.01

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


def seed_everything(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
