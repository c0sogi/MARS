import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Dual-Stream Hierarchical Recurrent Network (DS-HRN).
    Centralizes all file paths, hyperparameters, and global settings.
    """

    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Adjusted for 12 vCPUs
    DEBUG = False  # Set to True for quick debugging runs

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Directories
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
    SEGMENTATIONS_DIR = os.path.join(INPUT_DIR, "segmentations")

    # Metadata Files (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Auxiliary Data
    TRAIN_BBOXES_PATH = os.path.join(INPUT_DIR, "train_bounding_boxes.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directories (Output)
    WORKING_DIR = "./working/idea_9"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Preprocessing
    # =========================================================================
    # DICOM Windowing (Bone Window)
    WINDOW_CENTER = 400
    WINDOW_WIDTH = 1800

    # Image Dimensions
    IMG_SIZE_H = 512
    IMG_SIZE_W = 512

    # =========================================================================
    # Stage 1: Multi-Class Anatomical Localizer (2D U-Net)
    # =========================================================================
    SEG_MODEL_NAME = "unet"
    SEG_BACKBONE = "resnet18"
    SEG_IN_CHANS = 1
    SEG_NUM_CLASSES = 8  # Background (0) + C1-C7 (1-7)
    SEG_IMG_SIZE = (256, 256)  # Resized for speed

    # Training Hyperparameters
    SEG_BATCH_SIZE = 32
    SEG_EPOCHS = 15
    SEG_LR = 1e-4
    SEG_WEIGHT_DECAY = 1e-5

    # =========================================================================
    # Stage 2: Dual-Branch Feature Encoder
    # =========================================================================
    # Local Branch (High-Res Crop)
    LOCAL_CROP_SIZE = (224, 224)
    LOCAL_IN_CHANS = 4  # 3 (RGB/Repeated CT) + 1 (Binary Bone Mask)

    # Global Branch (Downsampled Full Slice)
    GLOBAL_SIZE = (224, 224)
    GLOBAL_IN_CHANS = 3  # 3 (RGB/Repeated CT)

    # Model Architecture
    ENC_BACKBONE = "tf_efficientnet_b0_ns"
    ENC_EMBED_DIM = 512  # Output feature dimension

    # Training Hyperparameters
    ENC_BATCH_SIZE = 32
    ENC_EPOCHS = 10
    ENC_LR = 1e-4
    ENC_WEIGHT_DECAY = 1e-5

    # =========================================================================
    # Stage 3: Hierarchical Anatomical Aggregator (Bi-GRU)
    # =========================================================================
    # Sequence Model
    RNN_HIDDEN_SIZE = 256
    RNN_LAYERS = 2
    RNN_DROPOUT = 0.2
    RNN_BIDIRECTIONAL = True

    # Input Dimension: Encoder Features + Anatomical Map (7 probs)
    RNN_INPUT_SIZE = ENC_EMBED_DIM + 7

    # Heads
    NUM_VERTEBRAE = 7

    # Training Hyperparameters
    RNN_BATCH_SIZE = 4  # Number of patients (sequences) per batch
    RNN_EPOCHS = 10
    RNN_LR = 1e-4
    RNN_WEIGHT_DECAY = 1e-4

    # =========================================================================
    # Utility Methods
    # =========================================================================
    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary output directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        cls.seed_everything()

    @classmethod
    def seed_everything(cls):
        """Sets the random seed for all relevant libraries."""
        random.seed(cls.SEED)
        os.environ["PYTHONHASHSEED"] = str(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        torch.cuda.manual_seed(cls.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
