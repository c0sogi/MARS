import os
import torch
import numpy as np
import random


class Config:
    """
    Global configuration for the Hybrid-Context Hierarchical Recurrent Network (HCH-RN).
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to use a subset of data for debugging
    DEBUG_SAMPLE_SIZE = 20  # Number of studies to use in debug mode

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Input Data Sources (Read-Only)
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
    SEGMENTATION_DIR = os.path.join(INPUT_DIR, "segmentations")

    # Metadata files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Auxiliary Data
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    TRAIN_BBOXES_PATH = os.path.join(INPUT_DIR, "train_bounding_boxes.csv")

    # Output / Cache Directories (Writeable)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    LOG_DIR = os.path.join(WORKING_DIR, "logs")

    # Specific Cache Files
    # Stage 1 Inference Results (ROI coords, Masks, Global Vectors)
    CACHE_SEG_INFERENCE_TRAIN = os.path.join(CACHE_DIR, "seg_inference_train.parquet")
    CACHE_SEG_INFERENCE_VAL = os.path.join(CACHE_DIR, "seg_inference_val.parquet")
    CACHE_SEG_INFERENCE_TEST = os.path.join(CACHE_DIR, "seg_inference_test.parquet")

    # Stage 2 Feature Embeddings (Input for Stage 3)
    CACHE_FEATURES_TRAIN = os.path.join(CACHE_DIR, "features_train.npy")
    CACHE_FEATURES_VAL = os.path.join(CACHE_DIR, "features_val.npy")
    CACHE_FEATURES_TEST = os.path.join(CACHE_DIR, "features_test.npy")

    # Final Submission
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Compute Configuration
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # -------------------------------------------------------------------------
    # Data Preprocessing
    # -------------------------------------------------------------------------
    # DICOM Windowing (Bone Window)
    WINDOW_CENTER = 400
    WINDOW_WIDTH = 1800

    # Image Dimensions
    ORIGINAL_SIZE = (512, 512)

    # Stage 1: Multi-Task Anatomical Segmentor Input
    IMG_SIZE_SEG = (256, 256)  # Downsampled for global context & segmentation

    # Stage 2: High-Resolution Encoder Input
    IMG_SIZE_CLS = (256, 256)  # Crops from original resolution

    # Stage 2 Input Channels: 3 consecutive slices (RGB) + 1 Mask = 4
    IN_CHANNELS_CLS = 4

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Stage 1: U-Net
    SEG_BACKBONE = "efficientnet_b0"
    SEG_ENCODER_WEIGHTS = "imagenet"
    SEG_CLASSES = 8  # Background + C1-C7

    # Stage 2: 2.5D CNN Encoder
    CLS_BACKBONE = "tf_efficientnetv2_s"
    CLS_PRETRAINED = True
    CLS_EMBED_DIM = 1280  # Output dimension of EfficientNetV2-S

    # Stage 3: Bi-GRU Aggregator
    RNN_HIDDEN_SIZE = 256
    RNN_LAYERS = 2
    RNN_DROPOUT = 0.2
    RNN_BIDIRECTIONAL = True
    # Input dim to RNN = Local Embed (1280) + Global Context (1280 from Stage 1) + Anatomical Prob (8)
    # Note: Global Context dim depends on SEG_BACKBONE (EffNetB0 is 1280)
    RNN_INPUT_DIM = 1280 + 1280 + 8

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Loss Weights (Competition Metric approximation)
    # patient_overall is weighted higher. Specific vertebrae are weighted lower.
    # Weights: {patient_overall: 7.0, C1-C7: 1.0} (Normalized or raw)
    LOSS_WEIGHTS = torch.tensor([1.0] * 7 + [7.0])  # C1..C7, patient_overall

    # Stage 1 Training (Segmentation)
    TRAIN_SEG_EPOCHS = 15
    TRAIN_SEG_BATCH_SIZE = 32
    TRAIN_SEG_LR = 1e-3

    # Stage 2 Training (Slice-level Classification)
    TRAIN_CLS_EPOCHS = 8
    TRAIN_CLS_BATCH_SIZE = 32
    TRAIN_CLS_LR = 1e-4

    # Stage 3 Training (Sequence Aggregation)
    TRAIN_RNN_EPOCHS = 10
    TRAIN_RNN_BATCH_SIZE = 4  # Small batch size due to variable sequence lengths
    TRAIN_RNN_LR = 5e-4

    # Optimization
    EARLY_STOPPING_PATIENCE = 3
    WEIGHT_DECAY = 1e-5

    @classmethod
    def setup(cls):
        """
        Initializes the environment: creates directories and sets random seeds.
        """
        # Create writeable directories
        directories = [
            cls.WORKING_DIR,
            cls.CACHE_DIR,
            cls.CHECKPOINT_DIR,
            cls.LOG_DIR,
            cls.SUBMISSION_DIR,
        ]
        for d in directories:
            os.makedirs(d, exist_ok=True)

        # Set Random Seeds for Reproducibility
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Configuration initialized. Working directory: {cls.WORKING_DIR}")
        print(f"Device: {cls.DEVICE}")
