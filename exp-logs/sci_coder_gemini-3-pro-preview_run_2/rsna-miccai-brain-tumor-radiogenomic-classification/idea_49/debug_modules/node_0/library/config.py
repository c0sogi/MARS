import os
import random
import numpy as np
import torch


class Config:
    """
    Central configuration for the Asymmetric Grouped EfficientNet pipeline.
    """

    # --------------------------------------------------------------------------
    # File Systems & Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    WORKING_DIR = "./working/idea_49"
    CACHE_DIR = WORKING_DIR
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing & ROI Selection
    # --------------------------------------------------------------------------
    IMG_SIZE = 224
    INTERPOLATION = "cv2.INTER_AREA"  # Recommended for downsampling to suppress noise

    # ROI Selection Constraints
    ROI_DEPTH_MIN = 0.15
    ROI_DEPTH_MAX = 0.85
    ROI_ANCHOR_MODALITY = "FLAIR"

    # Dual-Scale Input Strategy
    # Scales: Local (Stride 2) and Context (Stride 5)
    STRIDES = [2, 5]
    # Slices per scale: [Anchor-Stride, Anchor, Anchor+Stride]
    SLICES_PER_VIEW = 3

    # Input Dimensions
    NUM_MODALITIES = 4
    # Total Channels = 4 Modalities * 2 Scales * 3 Slices = 24
    IN_CHANNELS = NUM_MODALITIES * len(STRIDES) * SLICES_PER_VIEW

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    BACKBONE = "efficientnet_b0"
    # Groups = 8 (One group per Modality-Scale pair: 4 mods * 2 scales)
    # Each group processes 3 channels
    STEM_GROUPS = 8
    DROPOUT_RATE = 0.5

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 32  # Adjusted for A100 capacity
    EPOCHS = 20

    # Conservative Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    NUM_WORKERS = 4
    EARLY_STOPPING_PATIENCE = 5

    # --------------------------------------------------------------------------
    # Debugging / Runtime Control
    # --------------------------------------------------------------------------
    DEBUG = False
    MAX_DEBUG_SAMPLES = 50  # Limit dataset size when DEBUG is True

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


def set_seed(seed=42):
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Automatically setup directories when config is imported
Config.setup()
