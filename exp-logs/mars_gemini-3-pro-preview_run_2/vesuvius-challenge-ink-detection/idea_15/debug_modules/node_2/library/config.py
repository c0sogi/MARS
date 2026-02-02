import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the Vesuvius Ink Detection task.
    Implements parameters for the Hybrid SegFormer-UNet (MiT-B2) with Decoupled Inference Scanning.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache/output directory for this experimental idea
    IDEA_NAME = "idea_15"
    CACHE_DIR = os.path.join(WORKING_DIR, IDEA_NAME)
    CHECKPOINT_DIR = CACHE_DIR

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission Output
    SUBMISSION_PATH = "submission.csv"

    # Ensure working directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)

    # =========================================================================
    # Data Generation & Preprocessing
    # =========================================================================
    TILE_SIZE = 512
    STRIDE = 512  # Matches the non-overlapping stride used in metadata generation

    # Z-Slab Geometry (Overlapping Thick Slabs)
    # Training Context: Slices 20 to 44
    Z_START = 20
    MIP_DEPTH = 12  # Depth of each slab (channel)
    MIP_STRIDE = 6  # Overlap stride between channels
    MIP_CHANNELS = 3  # Number of input channels

    # Channel Mapping based on Z_START=20:
    # Ch 0: Slices 20-32
    # Ch 1: Slices 26-38
    # Ch 2: Slices 32-44

    # Normalization
    PIXEL_MIN = 0.0
    PIXEL_MAX = 65535.0  # uint16 range

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "hf_hub:nvidia/mit-b2"
    PRETRAINED = True
    IN_CHANNELS = 3
    NUM_CLASSES = 1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    EPOCHS = 15
    BATCH_SIZE = 8  # Conservative for 512x512 on A100 to ensure stability
    LEARNING_RATE = 6e-5
    NUM_WORKERS = 4

    # Validation & Checkpointing
    METRIC_BETA = 0.5  # F0.5 Score
    BASELINE_SCORE = 0.598  # Minimum score to beat to save submission

    # Debugging / Development
    DEBUG = False
    SAMPLE_SIZE = (
        None  # Set to an integer (e.g., 100) to limit dataset size for debugging
    )

    # =========================================================================
    # Inference Strategy (Decoupled Z-Scanning)
    # =========================================================================
    # Offsets applied to Z_START during inference to scan different depths.
    # -4 (Start 16), 0 (Start 20), +4 (Start 24)
    SCAN_OFFSETS = [-4, 0, 4]

    # Threshold for binary mask generation
    THRESHOLD = 0.5

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms can reduce performance, but ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Automatically set seed on import to ensure consistency
set_seed(Config.SEED)
