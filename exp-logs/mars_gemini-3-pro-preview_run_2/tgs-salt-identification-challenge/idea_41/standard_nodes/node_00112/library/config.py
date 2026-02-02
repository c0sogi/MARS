import os
import random
import numpy as np
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific Cache Directory for this Idea
    CACHE_DIR = "./working/idea_41"
    CHECKPOINT_DIR = os.path.join(CACHE_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    DEPTHS_CSV = os.path.join(INPUT_DIR, "depths.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    ORIG_SIZE = 101
    IMG_SIZE = 128  # Padded size (divisible by 32 for ResNet)
    CHANNELS = 1  # Seismic images are grayscale

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    FOLDS = 5
    BATCH_SIZE = 32

    # Epochs for each stage
    EPOCHS_TEACHER = 50
    EPOCHS_STUDENT = 50

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Debugging / Development
    DEBUG = False
    MAX_SAMPLES = None  # Set to integer to limit dataset size for debugging

    # Hardware
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Augmentation Parameters
    # =========================================================================
    # Elastic Transform (Crucial for salt)
    AUG_ELASTIC_ALPHA = 120.0
    AUG_ELASTIC_SIGMA = 6.0
    AUG_ELASTIC_ALPHA_AFFINE = 6.0
    AUG_ELASTIC_PROB = 0.2

    # Rigid (ShiftScaleRotate)
    AUG_RIGID_PROB = 0.2

    # =========================================================================
    # Model & Strategy Parameters
    # =========================================================================
    BACKBONE = "resnet34"
    ENCODER_PRETRAINED = "imagenet"

    # Marginalization: Depth scan values (z-scores)
    DEPTH_SCAN_SIGMAS = [-1.5, -0.75, 0.0, 0.75, 1.5]

    # Thresholding
    THRESHOLD_START = 0.5
    THRESHOLD_END = 0.95
    THRESHOLD_STEP = 0.05

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        cls.set_seed(cls.SEED)

    @staticmethod
    def set_seed(seed):
        """Sets random seeds for python, numpy, and torch."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
