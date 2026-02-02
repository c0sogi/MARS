import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging/testing

    # -------------------------------------------------------------------------
    # Hardware & Compute
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of data loading workers

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory (Read/Write) - used for caching and checkpoints
    # We use 'idea_3' as the specific workspace for this iteration
    WORKING_DIR = "./working/idea_3"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (for deterministic data processing)
    # These paths are used to store processed tiles/masks to speed up subsequent runs
    TRAIN_CACHE_DIR = os.path.join(WORKING_DIR, "train_cache")
    VAL_CACHE_DIR = os.path.join(WORKING_DIR, "val_cache")
    os.makedirs(TRAIN_CACHE_DIR, exist_ok=True)
    os.makedirs(VAL_CACHE_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    TILE_SIZE = 1024
    IN_CHANNELS = 4  # RGB (3) + Anatomical Mask (1)
    NUM_CLASSES = 1  # Binary segmentation (Glomerulus vs Background)

    # Sampling Strategy
    # Retain 100% of tiles with glomeruli, sample 20% of background-only tiles
    # This addresses the extreme sparsity of the target.
    BACKGROUND_SAMPLING_RATIO = 0.2

    # -------------------------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------------------------
    ENCODER_NAME = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    USE_ATTENTION = True  # Flag to enable Attention Gates in the U-Net decoder

    # -------------------------------------------------------------------------
    # Training Configuration
    # -------------------------------------------------------------------------
    BATCH_SIZE = 16  # Optimized for A100 40GB VRAM with 1024x1024 tiles
    EPOCHS = 60  # Total training epochs
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Warmup Strategy
    # Disable early stopping for the first 15 epochs to allow feature stabilization
    # given the small dataset size and high variance in validation metrics.
    WARMUP_EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement after warmup period

    # Loss Function Settings (Hybrid BCEDiceLoss)
    LOSS_BCE_WEIGHT = 0.5
    LOSS_DICE_WEIGHT = 0.5

    # -------------------------------------------------------------------------
    # Inference Configuration
    # -------------------------------------------------------------------------
    PREDICTION_THRESHOLD = 0.5
    OVERLAP_STRIDE = 0.5  # 50% overlap for sliding window inference
