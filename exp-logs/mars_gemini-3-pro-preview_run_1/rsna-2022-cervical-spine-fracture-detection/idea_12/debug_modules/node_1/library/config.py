import os
import torch
import random
import numpy as np


class Config:
    # ---------------------------------------------------------
    # 1. Paths & Directories
    # ---------------------------------------------------------
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"

    # Input Sub-directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
    SEGMENTATION_DIR = os.path.join(INPUT_DIR, "segmentations")

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    TRAIN_BBOX_PATH = os.path.join(INPUT_DIR, "train_bounding_boxes.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Directories (Created automatically)
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    LOG_DIR = os.path.join(WORKING_DIR, "logs")

    # ---------------------------------------------------------
    # 2. Global Settings
    # ---------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs
    DEBUG = False  # Set to True to run on a small subset for testing

    # ---------------------------------------------------------
    # 3. Data Preprocessing
    # ---------------------------------------------------------
    # DICOM Windowing (Bone Window)
    WINDOW_CENTER = 400
    WINDOW_WIDTH = 1800

    # Image Dimensions
    IMG_SIZE_ORIG = 512  # Original DICOM size
    IMG_SIZE_MODEL = 256  # Input size for CNNs (Global resize & Local crop)

    # Normalization
    PIXEL_MEAN = 0.456  # Approximate mean for bone windowed images
    PIXEL_STD = 0.224  # Approximate std

    # ---------------------------------------------------------
    # 4. Model Hyperparameters
    # ---------------------------------------------------------
    # Architecture
    BACKBONE = "efficientnet_b0"  # Lightweight backbone for speed
    IN_CHANNELS = 1  # Grayscale input (or +mask for local stream)

    # Stage 3: Aggregator
    GRU_HIDDEN_DIM = 256
    GCN_HIDDEN_DIM = 128
    NUM_VERTEBRAE = 7  # C1 to C7

    # ---------------------------------------------------------
    # 5. Training Hyperparameters
    # ---------------------------------------------------------
    # Stage 1: Segmentation / Localization
    BATCH_SIZE_SEG = 16
    LR_SEG = 1e-4
    EPOCHS_SEG = 10

    # Stage 2: Feature Encoder (Slice Classification)
    BATCH_SIZE_CLS = 32
    LR_CLS = 3e-4
    EPOCHS_CLS = 5

    # Stage 3: Sequence Aggregation
    BATCH_SIZE_SEQ = 4  # Patient-level batch size (sequence of slices)
    LR_SEQ = 5e-4
    EPOCHS_SEQ = 10

    # Optimization
    WEIGHT_DECAY = 1e-5
    PATIENCE = 3  # Early stopping patience

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        for d in [
            cls.WORKING_DIR,
            cls.CHECKPOINT_DIR,
            cls.CACHE_DIR,
            cls.SUBMISSION_DIR,
            cls.LOG_DIR,
        ]:
            os.makedirs(d, exist_ok=True)


def seed_everything(seed=42):
    """Sets the seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Initialize directories immediately upon import
Config.setup()
