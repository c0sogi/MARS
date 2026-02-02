import os
import torch
from pathlib import Path


class Config:
    """
    Central configuration for the Spatially Gated Dilated Network (SGDN) pipeline.
    Includes paths, hyperparameters, and constants derived from EDA and architectural design.
    """

    # --- File System Paths ---
    INPUT_DIR = Path("./input")
    METADATA_DIR = Path("./metadata")
    WORKING_DIR = Path("./working")

    # Cache directory specific to this experimental idea
    CACHE_DIR = WORKING_DIR / "idea_6"

    # Output path for the submission file
    SUBMISSION_PATH = Path("submission.csv")

    # Metadata file paths
    TRAIN_METADATA_PATH = METADATA_DIR / "train.csv"
    VAL_METADATA_PATH = METADATA_DIR / "val.csv"
    TEST_METADATA_PATH = METADATA_DIR / "test.csv"

    # --- Data Hyperparameters ---
    PATCH_SIZE = 256
    Z_DIM = 65  # Depth of the 3D volume

    # Normalization Statistics (Derived from EDA)
    # Used for global Z-score normalization
    PIXEL_MEAN = 99.9693
    PIXEL_STD = 12.5444

    # --- Model Architecture (SGDN) ---
    # Spatially Gated Dilated Network settings
    BASE_CHANNELS = 32
    MODEL_DEPTH = 8
    # Dilation rates sequence to manage receptive field without global pooling
    DILATION_RATES = [1, 2, 4, 8, 1, 2, 4, 8]
    # Group Normalization settings (groups=8 ensures stability with small/medium batches)
    GROUP_NORM_GROUPS = 8

    # --- Training Configuration ---
    SEED = 42
    BATCH_SIZE = 16  # Adjusted for A100 40GB
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    NUM_EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 5
    NUM_WORKERS = 4

    # Intensity Perturbation Augmentation
    # Randomly scales and shifts intensity to force structural learning over brightness
    INTENSITY_SCALE_RANGE = (0.8, 1.2)
    INTENSITY_OFFSET_RANGE = (-0.1, 0.1)  # Applied in normalized space

    # --- Inference & Evaluation ---
    # Stride for sliding window inference (50% overlap)
    INFERENCE_STRIDE = PATCH_SIZE // 2

    # Range of thresholds to scan for optimal F0.5 score
    THRESHOLD_SEARCH_RANGE = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # --- Debugging & Development ---
    # Set DEBUG to True to train on a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200  # Number of samples to use in debug mode

    # --- Compute ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.WORKING_DIR, exist_ok=True)


# Initialize directories on module import
Config.setup()
