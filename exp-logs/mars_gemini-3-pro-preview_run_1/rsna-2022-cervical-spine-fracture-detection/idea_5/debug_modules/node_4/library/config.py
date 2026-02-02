import os
import torch


class Config:
    """
    Central configuration for the Anatomically-Guided Transformer Pipeline.
    Handles paths, hyperparameters, and model settings for all three stages.
    """

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # Number of dataloader workers

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    # Input Data (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
    SEGMENTATION_DIR = os.path.join(INPUT_DIR, "segmentations")
    BBOX_CSV = os.path.join(INPUT_DIR, "train_bounding_boxes.csv")

    # Metadata (Pre-generated in ./metadata)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directory (Write Allowed)
    # Using 'idea_5' specific directory for caching and checkpoints
    WORKING_DIR = "./working/idea_5"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Stage 1: Anatomical Localizer & Segmentor (2D U-Net)
    # -------------------------------------------------------------------------
    # Purpose: Predict pixel-wise mask for C1-C7 to find center and anatomical ID.
    SEG_MODEL_NAME = "unet_resnet18"
    SEG_IMAGE_SIZE = (256, 256)  # Downsampled size for global context
    SEG_BATCH_SIZE = 32
    SEG_EPOCHS = 15
    SEG_LR = 1e-4
    SEG_NUM_CLASSES = 8  # 0=Background, 1-7=C1-C7

    # Checkpoint
    SEG_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "stage1_segmentor.pth")

    # Caching: Stores localization results (center coords + anatomical tags)
    TRAIN_SEG_CACHE = os.path.join(CACHE_DIR, "train_segmentation_meta.parquet")
    VAL_SEG_CACHE = os.path.join(CACHE_DIR, "val_segmentation_meta.parquet")
    TEST_SEG_CACHE = os.path.join(CACHE_DIR, "test_segmentation_meta.parquet")

    # -------------------------------------------------------------------------
    # Stage 2: Mask-Conditioned Focus Encoder (2.5D CNN)
    # -------------------------------------------------------------------------
    # Purpose: Extract features from high-res crops focused on the spine.
    # Input: 3 slices (RGB) + 1 Binary Bone Mask (Alpha) = 4 Channels
    ENC_BACKBONE = "tf_efficientnetv2_s"
    ENC_IMAGE_SIZE = (256, 256)  # High-res crop size
    ENC_IN_CHANNELS = 4
    ENC_SLICE_DEPTH = 3  # Number of consecutive slices used for 2.5D
    ENC_FEATURE_DIM = 1280  # Output dimension of the backbone

    ENC_BATCH_SIZE = 32
    ENC_EPOCHS = 5
    ENC_LR = 3e-4

    # Checkpoint
    ENC_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "stage2_encoder.pth")

    # Caching: Stores extracted feature vectors for sequence modeling
    TRAIN_FEATURES_CACHE = os.path.join(CACHE_DIR, "train_features.npy")
    VAL_FEATURES_CACHE = os.path.join(CACHE_DIR, "val_features.npy")
    TEST_FEATURES_CACHE = os.path.join(CACHE_DIR, "test_features.npy")

    # -------------------------------------------------------------------------
    # Stage 3: Anatomically-Embedded Transformer Aggregator
    # -------------------------------------------------------------------------
    # Purpose: Aggregate slice features + anatomical embeddings to predict fractures.
    AGG_HIDDEN_DIM = 512
    AGG_NUM_HEADS = 8
    AGG_NUM_LAYERS = 2
    AGG_DROPOUT = 0.1
    AGG_MAX_SEQ_LEN = 512  # Max slices per study (padded/truncated)

    AGG_BATCH_SIZE = 8  # Patient-level batch size
    AGG_EPOCHS = 10
    AGG_LR = 1e-4
    AGG_WEIGHT_DECAY = 1e-4

    # Checkpoint
    AGG_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "stage3_transformer.pth")

    # -------------------------------------------------------------------------
    # Data Processing Constants
    # -------------------------------------------------------------------------
    # DICOM Windowing (Bone Window)
    WINDOW_CENTER = 400
    WINDOW_WIDTH = 1800

    # Normalization
    PIXEL_MEAN = 0.5
    PIXEL_STD = 0.5
