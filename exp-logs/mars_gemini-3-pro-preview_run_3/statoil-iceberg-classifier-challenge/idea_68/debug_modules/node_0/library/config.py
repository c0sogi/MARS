import os
import torch


class Config:
    """
    Centralized configuration for the Robust-Texture Isomorphic CNN (RTI-CNN) solution.
    Defines paths, hyperparameters, and model architecture settings.
    """

    # =========================================================================
    # File Systems & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific experiment (Idea 68)
    WORKING_DIR = "./working/idea_68"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Metadata File Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data File Paths
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    IMG_HEIGHT = 75
    IMG_WIDTH = 75

    # Input channels: 3 (Band 1 HH, Band 2 HV, Average (HH+HV)/2)
    INPUT_CHANNELS = 3

    # Debugging / Quick Run parameters
    # Set DEBUG to True to run on a small subset for testing pipeline
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100

    # =========================================================================
    # Model Architecture (RTI-CNN)
    # =========================================================================
    # Backbone: Plain CNN (No Residuals)
    # Channel widths for the 4 sequential blocks
    BACKBONE_CHANNELS = [64, 128, 128, 128]

    # MAD Squeeze-and-Excitation settings
    SE_REDUCTION = 16  # Reduction ratio r=16

    # Activation settings
    LEAKY_RELU_SLOPE = 0.1

    # Readout & Classification Head
    # Projections reduce 128 channels to 64 before pooling
    PROJECTION_DIM = 64
    # Feature vector: Stage3 (Max+Min) + Stage4 (Max+Min) = 64*2 + 64*2 = 256
    FEATURE_DIM = 256
    DROPOUT_RATE = 0.5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    N_FOLDS = 5

    # Optimization
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3  # Constant LR
    WEIGHT_DECAY = 1e-2  # L2 Regularization for AdamW

    # Loop controls
    NUM_EPOCHS = 75
    PATIENCE = 12  # Early stopping patience

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates the necessary working directories if they do not exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def print_config(cls):
        """
        Prints the current configuration settings.
        """
        print(f"--- RTI-CNN Configuration ---")
        print(f"Device          : {cls.DEVICE}")
        print(f"Working Dir     : {cls.WORKING_DIR}")
        print(f"Input Channels  : {cls.INPUT_CHANNELS}")
        print(f"Backbone        : {cls.BACKBONE_CHANNELS}")
        print(f"Batch Size      : {cls.BATCH_SIZE}")
        print(f"Learning Rate   : {cls.LEARNING_RATE}")
        print(f"Weight Decay    : {cls.WEIGHT_DECAY}")
        print(f"Max Epochs      : {cls.NUM_EPOCHS}")
        print(f"Patience        : {cls.PATIENCE}")
        print(f"Seed            : {cls.SEED}")
        print(f"Debug Mode      : {cls.DEBUG}")
        print(f"-----------------------------")
