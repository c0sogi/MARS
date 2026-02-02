import os
import torch
import numpy as np


class Config:
    """
    Configuration for Salt Segmentation Task (Idea 26).
    Implements parameters for Corrected Multi-Task Wide-LinkNet with Ensemble Soft-Distillation.
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # 12 vCPUs available

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files (Generated previously)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data
    DEPTHS_CSV = os.path.join(INPUT_ROOT, "depths.csv")

    # Working Directory for Output
    WORKING_DIR = "./working/idea_26"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINTS_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Parameters
    # -------------------------------------------------------------------------
    # Original Image Dimensions
    ORIG_HEIGHT = 101
    ORIG_WIDTH = 101

    # Model Input Dimensions (Padded to be divisible by 32 for ResNet34)
    IMG_HEIGHT = 128
    IMG_WIDTH = 128

    # Input Channels
    # Modified to 1 channel (sum of RGB weights) as per strategy
    IN_CHANNELS = 1

    # Normalization (ImageNet Statistics)
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE = "resnet34"
    PRETRAINED = True

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Loss Configuration
    # Total Loss = Lovasz + BCE + (DEPTH_LOSS_WEIGHT * MSE)
    DEPTH_LOSS_WEIGHT = 0.1

    # -------------------------------------------------------------------------
    # Augmentation Parameters
    # -------------------------------------------------------------------------
    # Elastic Transform (Critical for organic salt structures)
    ELASTIC_ALPHA = 120.0
    ELASTIC_SIGMA = 6.0
    ELASTIC_ALPHA_AFFINE = 3.6  # approx 120 * 0.03

    # Rigid Transform (ShiftScaleRotate)
    RIGID_P = 0.2

    # -------------------------------------------------------------------------
    # Validation & Inference
    # -------------------------------------------------------------------------
    N_FOLDS = 5

    # IoU Thresholds for mAP calculation
    IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05).tolist()

    # Test Time Augmentation (Horizontal Flip)
    TTA = True

    # -------------------------------------------------------------------------
    # Debug / Development
    # -------------------------------------------------------------------------
    # If True, runs with a small subset of data for fewer epochs
    DEBUG = False

    @classmethod
    def set_debug_mode(cls, debug=True):
        """
        Helper to override config for debugging purposes.
        """
        cls.DEBUG = debug
        if debug:
            cls.EPOCHS = 2
            cls.BATCH_SIZE = 16
