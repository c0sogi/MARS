import os
import torch


class Config:
    """
    Centralized configuration for the Cactus Classification task.
    Implements settings for Metadata-Gated RepVGG with SWA training.
    """

    # -------------------------------------------------------------------------
    # General Setup & Reproducibility
    # -------------------------------------------------------------------------
    PROJECT_NAME = "cactus_classifier_idea_12"
    SEED = 42

    # Compute
    NUM_WORKERS = 4  # Optimized for the 12 vCPU environment
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Debugging
    # Set DEBUG to True to run on a small subset of data for pipeline verification
    DEBUG = False
    DEBUG_SAMPLES = 200

    # -------------------------------------------------------------------------
    # File System Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Caching Directory
    # Stores pre-processed tensors (images & file sizes) to eliminate I/O bottlenecks
    CACHE_DIR = "./working/idea_12"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Submission Path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Parameters
    # -------------------------------------------------------------------------
    IMAGE_SIZE = 32
    NUM_CLASSES = 2
    INPUT_CHANNELS = 3

    # Normalization Statistics (Derived from Dataset Analysis)
    # Values are converted from [0, 255] to [0, 1]
    # Mean: R=128.37, G=115.25, B=119.40
    # Std:  R=38.60,  G=35.68,  B=39.15
    NORM_MEAN = [0.5034, 0.4520, 0.4683]
    NORM_STD = [0.1514, 0.1399, 0.1535]

    # -------------------------------------------------------------------------
    # Model Architecture: Metadata-Gated RepVGG
    # -------------------------------------------------------------------------
    MODEL_NAME = "MetadataGatedRepVGG"

    # Backbone Configuration
    # Conservative downsampling strategy for 32x32 input
    BASE_WIDTH = 64
    # Stages: [Stage1_Blocks, Stage2_Blocks, Stage3_Blocks]
    # Resolution flow: 32x32 -> 16x16 -> 8x8 -> 4x4 (GAP)
    STAGE_DEPTHS = [2, 4, 6]
    STAGE_WIDTHS = [64, 128, 256]

    # Metadata Gating Head
    METADATA_DIM = 1  # Input dimension (Normalized File Size)
    GATE_HIDDEN_DIM = 64  # Hidden dimension for the gating MLP

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Stochastic Weight Averaging (SWA) Schedule
    EPOCHS_CONVERGENCE = 25  # Initial training phase
    EPOCHS_SWA = 10  # Exploration/Averaging phase
    TOTAL_EPOCHS = EPOCHS_CONVERGENCE + EPOCHS_SWA

    SWA_LR = 5e-4  # Learning rate for SWA phase
    SWA_START_EPOCH = EPOCHS_CONVERGENCE

    # Regularization
    MIXUP_ALPHA = 0.2  # Beta distribution parameter for Mixup

    # -------------------------------------------------------------------------
    # Inference Strategy
    # -------------------------------------------------------------------------
    # Test Time Augmentation (TTA)
    # 4 Views: Original, Horizontal Flip, Vertical Flip, Rotate 180
    TTA_VIEWS = 4
