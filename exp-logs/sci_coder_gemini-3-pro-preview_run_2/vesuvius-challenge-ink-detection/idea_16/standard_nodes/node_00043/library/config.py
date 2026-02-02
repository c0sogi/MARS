import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration for the Precision-Enhanced Hybrid SegFormer (MiT-B2) Pipeline.
    """

    # =========================================================================
    # General Setup
    # =========================================================================
    SEED = 42
    EXP_NAME = "idea_16"
    DEBUG = False  # Set to True to run on a small subset for debugging

    # =========================================================================
    # Compute Environment
    # =========================================================================
    NUM_WORKERS = 12  # Matches available vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Data & Preprocessing
    # =========================================================================
    TILE_SIZE = 512
    STRIDE = 256  # Overlap stride for inference tiling

    # Volumetric Strategy: Overlapping Stratified Depth Projection
    # We use 3 channels, each being a MIP of a thick slab.
    SLAB_THICKNESS = 12
    CHANNEL_OFFSETS = [0, 6, 12]  # Offsets relative to the scan start Z

    # Training Context: Fixed Narrow Context (slices 20-44)
    # Start at 20 -> Ch1: 20-32, Ch2: 26-38, Ch3: 32-44
    TRAIN_Z_START = 20

    # Inference Strategy: Decoupled Volumetric Z-Scanning
    # We scan multiple depths to capture wandering ink without polluting training data.
    # Scans: High (16), Center (20), Low (24)
    INFERENCE_Z_STARTS = [16, 20, 24]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Backbone: MiT-B2 (SegFormer)
    # Decoder: U-Net Style (to be implemented in model module)
    MODEL_ARCH = "mit_b2"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 3

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 15
    BATCH_SIZE = 8  # Fits on A100 with MiT-B2
    LEARNING_RATE = 6e-5
    WEIGHT_DECAY = 1e-2
    OPTIMIZER = "AdamW"
    LOSS_FUNCTION = "BCE_Dice"  # Binary Cross Entropy + Dice Loss

    # Validation Gating
    # Submission is only generated if Validation F0.5 > BASELINE_SCORE
    BASELINE_SCORE = 0.598

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for deterministic data processing (e.g., saved tensors)
    CACHE_DIR = os.path.join(WORKING_DIR, EXP_NAME)

    # Submission output path
    SUBMISSION_FILE = "submission.csv"

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VALIDATION_METADATA = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    @classmethod
    def get_channel_ranges(cls, start_z):
        """
        Calculates the slice ranges for the 3 input channels based on a starting Z-index.

        Args:
            start_z (int): The starting slice index for the scan.

        Returns:
            list of tuple: [(start1, end1), (start2, end2), (start3, end3)]
        """
        ranges = []
        for offset in cls.CHANNEL_OFFSETS:
            s = start_z + offset
            e = s + cls.SLAB_THICKNESS
            ranges.append((s, e))
        return ranges

    @staticmethod
    def set_seed(seed=42):
        """
        Sets fixed random seeds for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    @staticmethod
    def setup_directories():
        """
        Ensures necessary working directories exist.
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)


# Run setup on import to ensure directories exist
Config.setup_directories()
