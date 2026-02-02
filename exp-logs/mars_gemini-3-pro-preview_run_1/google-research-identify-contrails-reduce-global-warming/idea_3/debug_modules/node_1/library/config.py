import os
import torch


class Config:
    """
    Configuration for the Contrail Detection Task.
    Implements the 'Dilated ResNet-Ash U-Net' strategy.
    """

    # --------------------------------------------------------------------------
    # General Setup
    # --------------------------------------------------------------------------
    PROJECT_NAME = "idea_3"
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Utilizing available vCPUs (12 available)
    NUM_WORKERS = 12

    # --------------------------------------------------------------------------
    # Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = f"./working/{PROJECT_NAME}"

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VALIDATION_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINTS_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Data Preprocessing (Ash Composite)
    # --------------------------------------------------------------------------
    # Ash Composite Channels:
    # Red:   Band 15 - Band 14
    # Green: Band 14 - Band 11
    # Blue:  Band 14

    IMAGE_SIZE = 256

    # Wide Normalization Constants
    # Derived to prevent clipping of subtle signal tails.

    # Red Channel (Band 15 - Band 14)
    # Explicitly set based on statistical analysis of contrail thermal differentials
    ASH_RED_MIN = -6.7
    ASH_RED_MAX = 2.6

    # Green Channel (Band 14 - Band 11)
    # Widened range compared to standard heuristic (-4, 2) to capture full variance
    ASH_GREEN_MIN = -6.0
    ASH_GREEN_MAX = 6.0

    # Blue Channel (Band 14)
    # Brightness Temperature (Kelvin). Widened to cover full dynamic range.
    ASH_BLUE_MIN = 200.0
    ASH_BLUE_MAX = 320.0

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    ENCODER_NAME = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 3
    CLASSES = 1

    # Dilated ResNet Configuration
    # Use dilation in the final stage to maintain higher spatial resolution (16x downsampling)
    # instead of standard 32x.
    OUTPUT_STRIDE = 16
    REPLACE_STRIDE_WITH_DILATION = [False, False, True]

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 10

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500  # Number of samples to use when DEBUG is True

    # Loss Function Configuration (Focal + Dice)
    FOCAL_ALPHA = 0.5  # Balanced weighting for class imbalance
    FOCAL_GAMMA = 2.0
    FOCAL_WEIGHT = 0.5
    DICE_WEIGHT = 0.5

    # Optimizer & Scheduler
    SCHEDULER_PATIENCE = 2
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MIN_LR = 1e-7

    # --------------------------------------------------------------------------
    # Inference
    # --------------------------------------------------------------------------
    THRESHOLD = 0.5
    MIN_CONTRAIL_SIZE = 10  # Minimum pixels to be considered a contrail
