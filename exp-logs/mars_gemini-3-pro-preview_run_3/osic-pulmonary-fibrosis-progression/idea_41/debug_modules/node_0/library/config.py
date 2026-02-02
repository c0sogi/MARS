import os
import torch
import numpy as np
import random


class Config:
    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_DICOM_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DICOM_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directory for Idea 41
    WORKING_DIR = "./working/idea_41"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # -------------------------------------------------------------------------
    # Preprocessing & Data
    # -------------------------------------------------------------------------
    # EfficientNet-B2 native resolution
    IMG_SIZE = 260

    # Radiological Windowing (Lung Window)
    WINDOW_LEVEL = -600
    WINDOW_WIDTH = 1500

    # Slice Selection
    NUM_SLICES = 3  # Anchor + 2 boundaries

    # -------------------------------------------------------------------------
    # Model Architecture (PCDS-Net)
    # -------------------------------------------------------------------------
    BACKBONE_NAME = "efficientnet_b2"

    # Dimensionality
    BACKBONE_OUT_DIM = 1408  # EfficientNet-B2 output
    PROJECTION_DIM = 64  # Bottleneck projection
    CLINICAL_INPUT_DIM = 7  # Baseline FVC, Time, Age, Sex, Smoking(3)
    LATENT_DIM = 64  # Shared latent space dimension
    HIDDEN_DIM = 128  # MLP hidden layer size

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 30  # Can be overridden

    # Differential Learning Rates
    LR_BACKBONE = 1e-4
    LR_HEAD = 1e-3

    WEIGHT_DECAY = 1e-2

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
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
