import os
import torch
import numpy as np
import random


class Config:
    # =========================================================================
    # General Configuration
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
    SEGMENTATION_DIR = os.path.join(INPUT_DIR, "segmentations")

    # Metadata Paths
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Auxiliary Data
    TRAIN_BBOX_PATH = os.path.join(INPUT_DIR, "train_bounding_boxes.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directories (Write Access)
    # Main working directory for this specific idea/solution
    WORKING_DIR = "./working/idea_4"

    # Sub-directories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"  # Root submission dir as per requirement

    # =========================================================================
    # Data Preprocessing & Image Params
    # =========================================================================
    # DICOM Windowing (Bone Window)
    WINDOW_CENTER = 400
    WINDOW_WIDTH = 1800

    # Image Dimensions
    ORIGINAL_IMAGE_SIZE = 512
    # Crop size for the Stage 2 Encoder (High Res Crop)
    IMAGE_SIZE = 256

    # Stage 2 Input Configuration
    # Context: Number of neighbor slices to include.
    # 1 means: [slice-1, slice, slice+1] -> 3 channels
    NUM_SLICES_CONTEXT = 1
    TOTAL_SLICES_PER_INPUT = (NUM_SLICES_CONTEXT * 2) + 1  # 3 RGB channels
    USE_MASK_INPUT = True  # Add segmentation mask as 4th channel

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Stage 1: Segmentation (U-Net)
    SEG_MODEL_ARCH = "unet"
    SEG_BACKBONE = "resnet18"
    SEG_BATCH_SIZE = 16
    SEG_LR = 1e-4
    SEG_EPOCHS = 5

    # Stage 2: Slice Encoder (2.5D CNN)
    # Using a lightweight efficientnet for the slice encoder
    ENCODER_BACKBONE = "tf_efficientnetv2_s"
    ENCODER_IN_CHANNELS = 4 if USE_MASK_INPUT else 3
    ENCODER_FEATURE_DIM = 1280  # Depends on backbone
    ENCODER_HIDDEN_DIM = 256  # Projection dim
    CLS_BATCH_SIZE = 32
    CLS_LR = 1e-4
    CLS_EPOCHS = 5

    # Stage 3: Sequence Aggregator (Bi-GRU + Attention)
    SEQ_HIDDEN_DIM = 256
    SEQ_NUM_LAYERS = 2
    SEQ_DROPOUT = 0.2
    SEQ_BATCH_SIZE = 4  # Patient-level batch size (sequence of features)
    SEQ_LR = 5e-5
    SEQ_EPOCHS = 5

    # Max sequence length (slices per patient) for padding/truncating
    MAX_SEQ_LEN = 300

    # =========================================================================
    # Labels & Metrics
    # =========================================================================
    TARGET_COLS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    # Loss Weights: "The any label is weighted more highly"
    # We assign higher weight to patient_overall.
    # Format: {ColumnName: Weight}
    LOSS_WEIGHTS = {
        "C1": 1.0,
        "C2": 1.0,
        "C3": 1.0,
        "C4": 1.0,
        "C5": 1.0,
        "C6": 1.0,
        "C7": 1.0,
        "patient_overall": 7.0,
    }

    # =========================================================================
    # Setup Methods
    # =========================================================================
    @classmethod
    def setup(cls):
        """
        Initialize the environment: create directories and set random seeds.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set random seeds
        cls.seed_everything(cls.SEED)

    @staticmethod
    def seed_everything(seed):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
