import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration class for Cervical Spine Fracture Detection.
    Centralizes paths, hyperparameters, and hardware settings.
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 12  # Available vCPUs

    # Debugging Control
    # Set DEBUG to True to run on a small subset of data for quick testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific solution strategy
    WORKING_DIR = "./working/idea_3"

    # Input Data Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    TRAIN_BBOXES_PATH = os.path.join(INPUT_DIR, "train_bounding_boxes.csv")

    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
    SEGMENTATION_DIR = os.path.join(INPUT_DIR, "segmentations")

    # Output Paths
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    # CT Windowing (Bone Window)
    WINDOW_CENTER = 400
    WINDOW_WIDTH = 1800

    # Stage 1: Localizer (2D U-Net)
    # Input size for the spine localization network (downsampled context)
    LOCALIZER_IMG_SIZE = (256, 256)

    # Stage 2: Encoder (2.5D CNN)
    # High-resolution crop size centered on the spine
    ENCODER_CROP_SIZE = (256, 256)
    # Number of slices in the 2.5D stack (Center +/- 1)
    SLICES_IN_STACK = 3

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------

    # --- Phase 1: Localizer Training ---
    LOCALIZER_BACKBONE = "resnet18"
    LOCALIZER_BATCH_SIZE = 32
    LOCALIZER_EPOCHS = 15
    LOCALIZER_LR = 1e-4

    # --- Phase 2: Encoder Pre-training (Slice Level) ---
    ENCODER_BACKBONE = "resnet50"  # Or efficientnet_v2_s
    ENCODER_BATCH_SIZE = 64  # Adjust based on GPU VRAM (A100 40GB)
    ENCODER_EPOCHS = 5  # Short pre-training on slices
    ENCODER_LR = 1e-4

    # --- Phase 3: Aggregator Training (Patient Level) ---
    # Sequence model settings
    RNN_HIDDEN_SIZE = 256
    RNN_NUM_LAYERS = 2
    RNN_DROPOUT = 0.2

    # Batch size refers to number of patients (sequences)
    SEQ_BATCH_SIZE = 8
    SEQ_EPOCHS = 10
    SEQ_LR = 5e-4

    # Competition Metric Weights
    # patient_overall is weighted higher than individual vertebrae
    WEIGHT_PATIENT_OVERALL = 7.0
    WEIGHT_VERTEBRAE = 1.0

    @classmethod
    def setup(cls):
        """
        Initialize the environment: create directories and set random seeds.
        Must be called at the start of the pipeline.
        """
        # Create necessary directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducible seeds
        cls.seed_everything(cls.SEED)

    @staticmethod
    def seed_everything(seed):
        """Sets the seed for generating random numbers to ensure reproducibility."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
