import os
import torch


class Config:
    """
    Configuration for Contrail Identification Task.
    Implements the 'Large-Kernel ConvNeXt U-Net' strategy.
    """

    # ==========================
    # General Settings
    # ==========================
    PROJECT_NAME = "contrails_segmentation"
    IDEA_NAME = "idea_11"  # Strategy Identifier
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 500  # Number of samples if DEBUG is True

    # ==========================
    # Directories & Paths
    # ==========================
    # Input directory (Read-Only)
    INPUT_DIR = "./input"

    # Metadata paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VALID_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working directory for artifacts (Checkpoints, Cache)
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================
    # Data Configuration
    # ==========================
    IMG_SIZE = 256
    BATCH_SIZE = 32
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # Input Engineering
    # We use 6 channels:
    # Ch 1-3: Ash Color Scheme (Bands 11, 14, 15) at t=4
    # Ch 4-6: Temporal Difference (t=4 - t=3) for Bands 11, 14, 15
    IN_CHANNELS = 6

    # Temporal context provided in dataset
    N_TIMES_BEFORE = 4
    N_TIMES_AFTER = 3

    # ==========================
    # Model Architecture
    # ==========================
    # Backbone: ConvNeXt-Tiny (7x7 kernels for large receptive field)
    BACKBONE = "convnext_tiny"
    ENCODER_WEIGHTS = "imagenet"

    # Decoder: Custom Large-Kernel Decoder
    DECODER_BLOCK_TYPE = (
        "convnext"  # Use ConvNeXt blocks in decoder instead of standard Conv2d
    )
    DECODER_CHANNELS = [256, 128, 64, 32, 16]

    # ==========================
    # Training Hyperparameters
    # ==========================
    EPOCHS = 30
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01

    # Scheduler: Cosine Annealing
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Loss Function: Hybrid (BCE + BatchDice)
    BCE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5
    SMOOTH = 1e-6  # Smoothing factor for Dice calculation

    # ==========================
    # Inference & Post-processing
    # ==========================
    THRESHOLD = 0.5
    USE_TTA = True  # Test Time Augmentation (Horizontal Flip, Vertical Flip, Rotate)

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup_directories(cls):
        """
        Creates necessary directories for output and working files.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup_directories()
