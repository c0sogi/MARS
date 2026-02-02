import os
import torch
import random
import numpy as np


class Config:
    # =========================================================================
    # PATHS
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"

    # Input Subdirectories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
    SEGMENTATION_DIR = os.path.join(INPUT_DIR, "segmentations")
    TRAIN_BBOX_PATH = os.path.join(INPUT_DIR, "train_bounding_boxes.csv")

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Directories (created in setup)
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    PREDICTION_DIR = os.path.join(WORKING_DIR, "predictions")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    SUBMISSION_PATH = "./submission/submission.csv"

    # =========================================================================
    # DATA PREPROCESSING (DICOM & IMAGE)
    # =========================================================================
    # Bone Window Settings
    WINDOW_CENTER = 400
    WINDOW_WIDTH = 1800

    # Dimensions
    ORIGINAL_IMAGE_SIZE = 512
    # Stage 2 Input: Fixed-size crop centered on spine
    CROP_SIZE = 224

    # 2.5D Stacking: Number of slices to stack as channels (e.g., current, +1, -1)
    NUM_SLICES_25D = 3

    # =========================================================================
    # LABELS
    # =========================================================================
    TARGET_COLS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
    NUM_CLASSES = len(TARGET_COLS)

    # =========================================================================
    # MODEL HYPERPARAMETERS
    # =========================================================================
    # Stage 1: Localization
    SEG_MODEL_ARCH = "unet"
    SEG_BACKBONE = "resnet18"

    # Stage 2: Fracture Classification
    # Using a robust, efficient backbone for the 2.5D crop classifier
    CLS_MODEL_ARCH = "tf_efficientnet_v2_s"
    # Pretrained weights source (timm)
    PRETRAINED = True

    # =========================================================================
    # TRAINING HYPERPARAMETERS
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Matches available vCPUs

    # Batch Size & Optimization
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    MAX_GRAD_NORM = 10.0

    # Epochs
    NUM_EPOCHS = 10
    EARLY_STOPPING_PATIENCE = 3

    # Debugging / Development
    DEBUG = False
    DEBUG_DATASET_SIZE = 100  # Number of samples to use if DEBUG is True


def setup_directories():
    """
    Creates necessary working directories.
    """
    dirs = [
        Config.WORKING_DIR,
        Config.CHECKPOINT_DIR,
        Config.PREDICTION_DIR,
        Config.CACHE_DIR,
        os.path.dirname(Config.SUBMISSION_PATH),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
