import os
import torch
import random
import numpy as np


def seed_everything(seed=42):
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


class Config:
    # =========================================================================
    # System & Meta Configuration
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Matches available vCPUs

    # Debugging / Development
    # Set DEBUG to True to run on a small subset of data for quick testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 10

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    TRAIN_BOUNDING_BOXES_PATH = os.path.join(INPUT_DIR, "train_bounding_boxes.csv")

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
    SEGMENTATION_DIR = os.path.join(INPUT_DIR, "segmentations")

    # Working Directories (Write Allowed)
    WORKING_DIR = "./working/idea_11"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Preprocessing & Augmentation
    # =========================================================================
    # DICOM Windowing (Bone Window)
    WINDOW_CENTER = 400
    WINDOW_WIDTH = 1800

    # Image Dimensions
    IMAGE_SIZE_ORIGINAL = 512  # Original DICOM size
    IMAGE_SIZE_GLOBAL = 256  # Resized input for Global Stream (Branch B)
    IMAGE_SIZE_LOCAL = 256  # Crop size for Local Stream (Branch A)

    # Normalization (if used after windowing)
    PIXEL_MEAN = 0.5
    PIXEL_STD = 0.5

    # =========================================================================
    # Stage 1: Multi-Class Anatomical Localizer (2D U-Net)
    # =========================================================================
    STAGE1_MODEL_NAME = "unet_efficientnet_b0"
    STAGE1_LR = 1e-4
    STAGE1_BATCH_SIZE = 16
    STAGE1_EPOCHS = 10
    STAGE1_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "stage1_localizer.pth")

    # Classes: Background (0) + C1-C7 (1-7) = 8 classes
    STAGE1_NUM_CLASSES = 8

    # =========================================================================
    # Stage 2: Dual-Stream Feature Encoder
    # =========================================================================
    STAGE2_BACKBONE = "resnet18"  # Lightweight backbone for feature extraction
    STAGE2_LR = 3e-4
    STAGE2_BATCH_SIZE = 32
    STAGE2_EPOCHS = 5
    STAGE2_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "stage2_encoder.pth")

    # Input channels:
    # Local Stream: 1 (Image) + 1 (Mask) = 2 (or 3 if duplicated)
    # Global Stream: 1 (Image)
    # We typically adapt the first layer of the backbone.

    # =========================================================================
    # Stage 3: Anatomically-Indexed Recurrent Aggregator (Bi-GRU)
    # =========================================================================
    STAGE3_HIDDEN_DIM = 256
    STAGE3_NUM_LAYERS = 2
    STAGE3_DROPOUT = 0.2
    STAGE3_LR = 5e-4
    STAGE3_BATCH_SIZE = 4  # Batch size is number of patients (sequences)
    STAGE3_EPOCHS = 5
    STAGE3_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "stage3_aggregator.pth")

    # Targets
    TARGET_COLS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    # Loss Weights (Competition Metric)
    # patient_overall is weighted higher. Specific values can be tuned or derived.
    # Standard competition weights: 1 for vertebrae, higher for patient.
    # Here we define the column indices for reference.
