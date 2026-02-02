import os
import numpy as np
import torch


class Config:
    """
    Configuration for Salt Segmentation Task: Multi-Task Depth-Regularized Noisy Student.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False
    NUM_WORKERS = 12  # Utilizing all available vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_25"

    # Create necessary working subdirectories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    DEPTHS_CSV = os.path.join(INPUT_DIR, "depths.csv")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    # Dimensions
    ORIG_SIZE = 101
    IMG_SIZE = 128  # Pad to 128x128 for model divisibility (32)
    CHANNELS = 1  # Grayscale input adaptation

    # Normalization (Derived from EDA: Global Mean ~148, Std ~65 on 0-255 scale)
    # Normalized to [0, 1]: Mean ~0.58, Std ~0.25
    PIXEL_MEAN = [0.58]
    PIXEL_STD = [0.25]

    # Depth Normalization (Approximate from range [51, 959])
    DEPTH_MEAN = 505.0
    DEPTH_STD = 250.0

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    ENCODER_NAME = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    DECODER_CHANNELS = [256, 128, 64, 32, 16]

    # Multi-Task / Injection Settings
    INJECT_DEPTH = True
    AUX_DEPTH_HEAD = True

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 64  # A100 40GB allows for larger batches
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Stage Durations
    EPOCHS_STAGE1 = 50  # Supervised Multi-Task
    EPOCHS_STAGE2 = 1  # Pseudo-label generation (inference only)
    EPOCHS_STAGE3 = 50  # Noisy Student

    # Loss Configuration
    # Loss = (Lovasz + BCE) + DEPTH_LOSS_WEIGHT * MSE_Depth
    DEPTH_LOSS_WEIGHT = 0.1

    # Bernoulli Injection Masking
    # Probability of replacing True Depth with Mean Depth (0 after scaling) during training
    BERNOULLI_DROP_PROB = 0.5

    # =========================================================================
    # Augmentation Parameters
    # =========================================================================
    # Mandatory Non-Rigid (Elastic)
    ELASTIC_ALPHA = 120
    ELASTIC_SIGMA = 6
    ELASTIC_PROB = 1.0

    # Rigid (ShiftScaleRotate)
    RIGID_AUG_PROB = 0.2

    # =========================================================================
    # Metric & Evaluation
    # =========================================================================
    # IoU Thresholds: 0.5 to 0.95 with step 0.05
    IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)

    # Test Time Augmentation
    TTA_STEPS = 2  # Original + Horizontal Flip

    @classmethod
    def setup(cls, debug=False):
        """
        Sets up the configuration, creating directories and adjusting for debug mode.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        cls.DEBUG = debug
        if cls.DEBUG:
            print("(!) DEBUG MODE ENABLED: Reducing epochs and data usage.")
            cls.EPOCHS_STAGE1 = 2
            cls.EPOCHS_STAGE3 = 2
            cls.BATCH_SIZE = 16
            # In debug mode, data loaders should sample a subset,
            # this flag can be checked by the dataset class.
