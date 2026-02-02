import os
import torch


class Config:
    """
    Configuration class for the 3D Stomach and Intestines Segmentation pipeline.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # ==============================
    # General Settings
    # ==============================
    SEED = 42
    DEBUG = False  # Set to True to run on a subset of data for quick testing
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Optimized for the 12 vCPU environment

    # ==============================
    # Directory Paths
    # ==============================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working Directories (Write Access)
    WORKING_DIR = "./working/idea_6"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    PREDICTION_DIR = os.path.join(WORKING_DIR, "predictions")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # ==============================
    # Data Preprocessing & Augmentation
    # ==============================
    # 3D Patch Size: (Depth, Height, Width)
    # Selected based on A100 40GB memory constraints and 3D ResNet requirements
    PATCH_SIZE = (32, 224, 224)
    SPATIAL_SIZE = (224, 224)
    DEPTH_SIZE = 32

    # Robust Percentile Normalization
    # Clips intensities to exclude outliers before scaling to [0, 1]
    LOWER_PERCENTILE = 1.0
    UPPER_PERCENTILE = 99.0

    # Target Classes
    CLASSES = ["large_bowel", "small_bowel", "stomach"]
    NUM_CLASSES = 3

    # ==============================
    # Model Architecture
    # ==============================
    # Using 3D ResNet-18 as the encoder backbone
    BACKBONE = "r3d_18"
    IN_CHANNELS = 1
    OUT_CHANNELS = 3

    # ==============================
    # Training Hyperparameters
    # ==============================
    BATCH_SIZE = 4  # Conservative batch size for 3D volumes
    EPOCHS = 20  # Sufficient for convergence given the pre-trained backbone
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5

    # Loss Weights
    DICE_WEIGHT = 0.4
    HAUSDORFF_WEIGHT = 0.6  # Indirectly optimized via Dice + topology checks

    # ==============================
    # Inference Strategy
    # ==============================
    # Sliding Window Inference settings
    ROI_SIZE = PATCH_SIZE
    SW_BATCH_SIZE = 4
    OVERLAP = 0.5  # 50% overlap to mitigate boundary artifacts

    # Post-processing
    # Minimum volume to keep a connected component (to remove noise)
    MIN_COMPONENT_SIZE = 100

    @classmethod
    def setup(cls):
        """
        Creates the necessary directory structure in the working directory.
        Should be called at the start of the pipeline.
        """
        dirs = [
            cls.WORKING_DIR,
            cls.CHECKPOINT_DIR,
            cls.PREDICTION_DIR,
            cls.SUBMISSION_DIR,
            cls.CACHE_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        print(f"Configuration setup complete. Working directory: {cls.WORKING_DIR}")

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k}: {v}")
        print("=" * 30)
