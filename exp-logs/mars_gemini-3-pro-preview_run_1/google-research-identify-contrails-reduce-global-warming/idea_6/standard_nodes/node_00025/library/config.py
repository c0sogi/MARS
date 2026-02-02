import os
import torch
import numpy as np
import random


class AshConfig:
    """
    Configuration for Ash False-Color Composite generation.
    Defines the specific bands and normalization bounds to create
    high-contrast inputs for the model.
    """

    # Band definitions (GOES-16 ABI Band Numbers)
    # Red: Optical Depth Proxy (Band 15 - Band 14)
    # Green: Particle Phase Proxy (Band 14 - Band 11)
    # Blue: Temperature (Band 14)
    BAND_RED_1 = 15
    BAND_RED_2 = 14
    BAND_GREEN_1 = 14
    BAND_GREEN_2 = 11
    BAND_BLUE = 14

    # Normalization Bounds (Kelvin)
    # These "Standard" bounds clip background and outliers to maximize contrail contrast.
    # Red Channel (Difference)
    RED_MIN = -4.0
    RED_MAX = 2.0

    # Green Channel (Difference)
    GREEN_MIN = -4.0
    GREEN_MAX = 2.0

    # Blue Channel (Temperature)
    BLUE_MIN = 243.0
    BLUE_MAX = 303.0


class Config:
    """
    Global configuration for the Contrail Identification pipeline.
    """

    # --------------------------------------------------------------------------
    # System & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------------------
    # File Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea/experiment
    WORKING_DIR = "./working/idea_6"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    PREDICTION_DIR = os.path.join(WORKING_DIR, "predictions")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # --------------------------------------------------------------------------
    # Data Parameters
    # --------------------------------------------------------------------------
    IMG_SIZE = 256
    # Sequence details: 4 frames before, 1 labeled frame, 3 frames after
    N_TIMES_BEFORE = 4
    N_TIMES_AFTER = 3
    TOTAL_FRAMES = N_TIMES_BEFORE + N_TIMES_AFTER + 1

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    MODEL_ARCH = "UnetPlusPlus"  # Nested U-Net
    ENCODER_NAME = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 3
    CLASSES = 1
    ACTIVATION = None  # Logits for numerical stability in loss

    # Dilated convolutions in the final encoder stage to preserve spatial resolution
    # Standard ResNet reduces to 1/32, we want 1/16 max reduction
    ENCODER_OUTPUT_STRIDE = 16

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 35
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01

    # Loss Configuration
    # Weighted composite: 0.75 * Focal + 0.25 * Dice
    WEIGHT_FOCAL = 0.75
    WEIGHT_DICE = 0.25

    # Focal Loss Params
    FOCAL_ALPHA = 0.75  # Prioritize positive class
    FOCAL_GAMMA = 2.0

    # Scheduler
    # Reduce LR when validation Dice score stagnates
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MIN_LR = 1e-6

    # --------------------------------------------------------------------------
    # Inference
    # --------------------------------------------------------------------------
    THRESHOLD = 0.5

    @staticmethod
    def setup():
        """
        Initializes necessary directories and sets random seeds.
        """
        # Create directories
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.PREDICTION_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

        # Set seeds
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        torch.cuda.manual_seed(Config.SEED)

        # Deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Execute setup immediately upon import to ensure environment is ready
Config.setup()
