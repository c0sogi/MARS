import os
import torch
import numpy as np
import random


class Config:
    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for specific idea (idea_2 based on prompt requirements)
    WORKING_DIR = "./working/idea_2"

    # Cache directory for preprocessed features
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Directory to save model checkpoints
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Directory for final submission
    SUBMISSION_DIR = "./submission"

    # Metadata files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Data & Features
    # =========================================================================
    # Gesture vocabulary: 20 classes + 1 background class (index 0)
    # Labels in data are 1-20. We map them to 1-20 in model, 0 is background.
    NUM_CLASSES = 21

    # Feature Engineering Flags
    USE_SKELETON = True
    USE_VELOCITY = True  # Compute first derivative of skeleton joints
    USE_AUDIO = True  # Extract MFCCs

    # Skeleton params
    NUM_JOINTS = 20
    COORDS_PER_JOINT = 3  # X, Y, Z

    # Audio params
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_N_MFCC = 13
    # Video FPS (approximate, used for aligning audio)
    VIDEO_FPS = 10

    # Input Dimension Calculation
    # Skeleton (20*3) + Velocity (20*3) + Audio (13) = 60 + 60 + 13 = 133
    INPUT_DIM = (NUM_JOINTS * COORDS_PER_JOINT) * (1 + int(USE_VELOCITY)) + (
        AUDIO_N_MFCC if USE_AUDIO else 0
    )

    # =========================================================================
    # Model Architecture (MS-TCN)
    # =========================================================================
    # Number of stages (Prediction + Refinement stages)
    NUM_STAGES = 4

    # Number of layers per stage
    # Dilations will be 2^0, 2^1, ..., 2^(NUM_LAYERS-1)
    NUM_LAYERS = 10

    # Number of feature maps (channels) in internal layers
    NUM_F_MAPS = 64

    # Kernel size for dilated convolutions
    KERNEL_SIZE = 3

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-4
    NUM_EPOCHS = 50

    # Loss Weights
    # Weight for the smoothing loss (T-MSE)
    LAMBDA_SMOOTHING = 0.15

    # Class weighting strategy
    # Since background (0) is dominant, we might weight it less or others more.
    # This can be computed dynamically or set here.
    # For now, we enable weighted loss flag.
    USE_WEIGHTED_LOSS = True

    # Early Stopping
    PATIENCE = 10

    # =========================================================================
    # Post-Processing
    # =========================================================================
    # Window size for median filtering of predictions
    MEDIAN_WINDOW_SIZE = 7

    @classmethod
    def setup(cls):
        """
        Creates necessary directories and sets random seeds.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Initialize setup immediately when module is imported to ensure consistency
Config.setup()
