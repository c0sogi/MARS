import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Contrail Identification task.
    Implements settings for the Attention-Gated Large-Kernel ConvNeXt U-Net strategy.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata CSV Paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VALIDATION_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Directories
    # Working directory for checkpoints and cached data
    WORKING_DIR = "./working/idea_18"
    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMAGE_SIZE = 256

    # Temporal sequence details
    # n_times_before=4, labeled_frame, n_times_after=3 -> Total 8 frames
    NUM_FRAMES = 8
    LABELED_FRAME_IDX = 4  # The 5th frame (index 4) is the target
    PREV_FRAME_IDX = 3  # The 4th frame (index 3) is used for temporal difference

    # Input Channels
    # Channels 1-3: Ash False Color Composite
    # Channels 4-6: Raw Band Differences (t=4 - t=3) for Bands 11, 14, 15
    IN_CHANNELS = 6
    NUM_CLASSES = 1

    # Ash Color Scheme Normalization Constants
    # Derived from standard GOES-16 Ash RGB recipes and domain heuristics
    # Red: Band 15 - Band 13
    ASH_RED_MIN = -6.7
    ASH_RED_MAX = 2.6

    # Green: Band 14 - Band 11
    ASH_GREEN_MIN = -6.0
    ASH_GREEN_MAX = 6.0

    # Blue: Band 13 (Brightness Temperature)
    ASH_BLUE_MIN = 240.0
    ASH_BLUE_MAX = 300.0

    # Bands used for the temporal difference channels (Channels 4-6)
    DIFF_BANDS = [11, 14, 15]

    # ==========================================
    # Model Configuration
    # ==========================================
    # Backbone: ConvNeXt-Tiny with 7x7 kernels for linearity prior
    ENCODER_NAME = "convnext_tiny"
    ENCODER_WEIGHTS = "imagenet"

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 30

    # Optimization
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2

    # Compute
    NUM_WORKERS = 4  # Optimized for 12 vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Debugging & Development
    # ==========================================
    # Controls for dataset size to allow fast debugging cycles
    # Set to an integer (e.g., 1000) to limit the number of samples
    MAX_TRAIN_SAMPLES = None
    MAX_VAL_SAMPLES = None

    # Threshold for post-processing
    THRESHOLD = 0.5

    @staticmethod
    def set_seed(seed=42):
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @classmethod
    def setup_directories(cls):
        """
        Ensures that necessary working and submission directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
