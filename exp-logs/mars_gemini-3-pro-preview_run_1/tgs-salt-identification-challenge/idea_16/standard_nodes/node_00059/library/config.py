import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration for Salt Segmentation Task.
    Strategy: Deep Residual U-Net with Homogeneous Lovasz-Cycle Ensembling.
    """

    # -------------------------------------------------------------------------
    # 1. Paths and Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_16"

    # Sub-directories for artifacts
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    CACHE_DIR = WORKING_DIR  # Directory for caching processed data

    # Metadata File Paths
    # Note: Using provided metadata which contains 80/20 split.
    # Strategy mentions 90/10 but we adhere to provided environment files for consistency.
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data Paths
    DEPTHS_CSV = os.path.join(INPUT_DIR, "depths.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # -------------------------------------------------------------------------
    # 2. Data Configuration
    # -------------------------------------------------------------------------
    # Original Image Dimensions
    ORIG_HEIGHT = 101
    ORIG_WIDTH = 101

    # Model Input Dimensions (Padded with Reflection Padding)
    IMG_HEIGHT = 128
    IMG_WIDTH = 128

    # Channel Configuration
    IMG_CHANNELS = 1  # Grayscale Seismic Image
    DEPTH_CHANNELS = 1  # Depth Map (Dense Channel)
    INPUT_CHANNELS = 2  # Total: Image + Depth

    # Data Loading
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # -------------------------------------------------------------------------
    # 3. Model Configuration
    # -------------------------------------------------------------------------
    # Architecture: Deep Residual U-Net with scSE
    # Encoder: ResNet-style blocks
    ENCODER_FILTERS = [64, 128, 256, 512]  # Filters at each downsampling step

    # Decoder: Symmetric upsampling
    DECODER_FILTERS = [256, 128, 64, 32]

    # Attention Mechanism
    USE_SCSE = True  # Concurrent Spatial and Channel Squeeze & Excitation

    # Deep Supervision
    USE_DEEP_SUPERVISION = True
    DS_RESOLUTIONS = [32, 64, 128]  # Resolutions for auxiliary heads

    # -------------------------------------------------------------------------
    # 4. Training Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler: Cosine Annealing Warm Restarts
    TOTAL_EPOCHS = 150
    CYCLES = 3
    EPOCHS_PER_CYCLE = 50

    # Loss Function Curriculum
    # Cycle 1 (0-50): Establish structure with BCE + Dice
    # Cycles 2 & 3 (51-150): Optimize metric with BCE + Lovasz-Hinge
    CYCLE_1_END_EPOCH = 50

    # Snapshot Ensembling
    # We save the best model from Cycle 2 and Cycle 3 for ensembling
    SAVE_CYCLES = [2, 3]

    # -------------------------------------------------------------------------
    # 5. Inference Configuration
    # -------------------------------------------------------------------------
    # Test-Time Augmentation
    TTA_FLIP = True  # Average predictions with horizontally flipped version

    # Metric Calculation
    IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    @staticmethod
    def setup():
        """Creates necessary working directories and sets random seeds."""
        # Create directories
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Set reproducible seeds
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(Config.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
