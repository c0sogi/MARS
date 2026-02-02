import os
import torch


class Config:
    """
    Global configuration for the Asymmetric Grouped EfficientNet with
    Raw-Selected Independent-Norm Pipeline.

    This configuration centralizes all hyperparameters for data ingestion,
    model architecture, and training loops to ensure reproducibility and
    strict adherence to the proposed strategy.
    """

    # --------------------------------------------------------------------------
    # System & Paths
    # --------------------------------------------------------------------------
    # Base directories
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output & Working Directory
    # Using 'idea_22' to isolate this specific architectural iteration
    WORKING_DIR = "./working/idea_22"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Artifacts
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Caching
    # Used to store expensive ROI calculations
    CACHE_DIR = WORKING_DIR
    ROI_CACHE_FILE = os.path.join(CACHE_DIR, "roi_cache.parquet")

    # --------------------------------------------------------------------------
    # Data Pipeline Hyperparameters
    # --------------------------------------------------------------------------
    IMG_SIZE = 224

    # Stacking Logic
    # We stack 3 slices from 4 modalities = 12 Channels
    NUM_MODALITIES = 4  # FLAIR, T1w, T1wCE, T2w
    NUM_SLICES = 3  # Anchor + 2 neighbors
    STRIDE = 5  # Fixed stride for neighbor selection (Idea 21 spec)
    IN_CHANNELS = 12  # 4 * 3

    # ROI Selection (Raw-Selected)
    # Restrict search to 15%-85% depth to avoid skull/neck artifacts
    ROI_DEPTH_MIN = 0.15
    ROI_DEPTH_MAX = 0.85
    ROI_ANCHOR_MODALITY = "FLAIR"

    # Normalization
    # Independent per-channel normalization to [0, 1] to preserve dynamic range
    NORM_METHOD = "independent_slice"

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True
    NUM_CLASSES = 1
    DROPOUT_RATE = 0.3  # Standard for EfficientNet

    # Stem Modifications for 12-channel input
    STEM_GROUPS = 4  # Enforce modality isolation in first layer
    ASYMMETRIC_INIT = True  # Distribute ImageNet filters across groups

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 15  # Conservative epoch count for fine-tuning

    # Optimization
    LEARNING_RATE = 1e-4  # Low LR to preserve pre-trained features
    WEIGHT_DECAY = 1e-2  # Aggressive weight decay
    PATIENCE = 5  # Early stopping patience

    # Augmentation
    ROTATION_DEGREES = 15  # +/- 15 degrees

    # Hardware
    NUM_WORKERS = 4  # Optimized for 12 vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
