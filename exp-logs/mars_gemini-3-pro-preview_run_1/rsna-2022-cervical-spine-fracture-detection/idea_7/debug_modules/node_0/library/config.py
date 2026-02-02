import os
import torch
import numpy as np
import random


class Config:
    # --------------------------------------------------------------------------
    # General Settings
    # --------------------------------------------------------------------------
    PROJECT_NAME = "idea_7_dual_res_rnn"
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use fewer workers if debugging to avoid overhead, otherwise use CPU count
    NUM_WORKERS = 2 if "DEBUG" in os.environ else os.cpu_count()

    # Debugging Control
    # Set to True to run on a small subset of data for pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 20

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
    SEGMENTATIONS_DIR = os.path.join(INPUT_DIR, "segmentations")

    # Metadata Directories (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    BBOX_PATH = os.path.join(INPUT_DIR, "train_bounding_boxes.csv")

    # Working Directories (Write Access)
    # Specific to Idea 7
    WORKING_DIR = "./working/idea_7"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    LOG_DIR = os.path.join(WORKING_DIR, "logs")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Preprocessing & Augmentation
    # --------------------------------------------------------------------------
    # DICOM Windowing (Bone Window)
    WINDOW_CENTER = 400
    WINDOW_WIDTH = 1800

    # Image Dimensions
    ORIGINAL_SIZE = (512, 512)

    # Stage 1: Global Context & Localization (Downsampled)
    STAGE1_IMAGE_SIZE = (256, 256)

    # Stage 2: Local Fracture Stream (High-Res Crops)
    STAGE2_CROP_SIZE = (256, 256)
    STAGE2_INPUT_DEPTH = 3  # Number of slices in stack (RGB-like)
    STAGE2_USE_MASK = True  # Append segmentation mask as 4th channel

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # --------------------------------------------------------------------------

    # Stage 1: U-Net (Segmentation & Global Context)
    STAGE1_BACKBONE = "resnet18"
    STAGE1_IN_CHANNELS = 1  # Grayscale CT
    STAGE1_NUM_CLASSES = 8  # Background (0) + C1-C7 (1-7)

    # Stage 2: 2.5D CNN (Fracture Classification on Crops)
    STAGE2_BACKBONE = "efficientnet_v2_s"
    STAGE2_IN_CHANNELS = 4  # 3 slices + 1 mask
    STAGE2_EMBEDDING_DIM = 512  # Output dimension of the CNN encoder

    # Stage 3: Bi-GRU (Sequence Aggregation)
    STAGE3_RNN_HIDDEN_SIZE = 256
    STAGE3_RNN_LAYERS = 2
    STAGE3_DROPOUT = 0.2
    # Output heads: 7 for vertebrae (C1-C7) + 1 for patient_overall
    STAGE3_NUM_CLASSES = 8

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------

    # Stage 1: Segmentation Training
    STAGE1_BATCH_SIZE = 32
    STAGE1_EPOCHS = 10
    STAGE1_LR = 1e-4
    STAGE1_WEIGHT_DECAY = 1e-5

    # Stage 2: Slice/Crop Classification Training
    STAGE2_BATCH_SIZE = 32
    STAGE2_EPOCHS = 5
    STAGE2_LR = 1e-4
    STAGE2_WEIGHT_DECAY = 1e-5

    # Stage 3: Sequence Aggregation Training
    STAGE3_BATCH_SIZE = 8  # Batch size in terms of Patients/Studies
    STAGE3_EPOCHS = 10
    STAGE3_LR = 5e-4
    STAGE3_WEIGHT_DECAY = 1e-4

    # Inference
    TEST_BATCH_SIZE = 1

    # --------------------------------------------------------------------------
    # Utility Methods
    # --------------------------------------------------------------------------
    @staticmethod
    def setup(make_dirs=True):
        """
        Initializes the environment:
        1. Sets random seeds for reproducibility.
        2. Creates necessary working directories.
        """
        # 1. Fix Seeds
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # 2. Create Directories
        if make_dirs:
            for directory in [
                Config.WORKING_DIR,
                Config.CACHE_DIR,
                Config.CHECKPOINT_DIR,
                Config.LOG_DIR,
                Config.SUBMISSION_DIR,
            ]:
                os.makedirs(directory, exist_ok=True)

    @staticmethod
    def print_config():
        """Prints the current configuration setup."""
        print(f"\n=== Configuration: {Config.PROJECT_NAME} ===")
        print(f"Device      : {Config.DEVICE}")
        print(f"Seed        : {Config.SEED}")
        print(f"Working Dir : {Config.WORKING_DIR}")
        print(f"Debug Mode  : {Config.DEBUG}")
        if Config.DEBUG:
            print(f"Debug Sample: {Config.DEBUG_SAMPLE_SIZE}")
        print("==========================================\n")
