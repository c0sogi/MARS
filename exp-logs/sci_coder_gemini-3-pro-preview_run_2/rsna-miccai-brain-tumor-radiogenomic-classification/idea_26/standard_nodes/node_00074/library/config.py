import os
import torch


class Config:
    """
    Configuration class for the Asymmetric Grouped EfficientNet pipeline.
    Acts as the single source of truth for hyperparameters, paths, and settings.
    """

    # --------------------------------------------------------------------------
    # System & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Specific working directory for Idea 26 (Asymmetric Grouped EfficientNet)
    WORKING_DIR = "./working/idea_26"
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Cache Paths (for deterministic data processing)
    # Used by data processing modules to store/load ROI indices or processed arrays
    CACHE_ROI_PATH = os.path.join(WORKING_DIR, "roi_cache.parquet")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Data Processing Parameters
    # --------------------------------------------------------------------------
    IMG_SIZE = 224

    # Modalities to use (Order matters for channel interleaving)
    MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]

    # ROI Selection (Raw-Integral based)
    ROI_MODALITY = "FLAIR"
    ROI_BOUNDS = [0.15, 0.85]  # Restrict search to 15% - 85% depth

    # Dual-Stride Interleaved Stacking Logic
    # We extract a "Local" context (Stride 2) and a "Global" context (Stride 10)
    # Each stride group consists of 3 slices: [Anchor - Stride, Anchor, Anchor + Stride]
    STRIDES = [2, 10]
    SLICES_PER_STRIDE = 3

    # Input Channel Calculation:
    # 4 Modalities * 2 Stride Groups * 3 Slices/Group = 24 Channels
    INPUT_CHANNELS = len(MODALITIES) * len(STRIDES) * SLICES_PER_STRIDE

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True

    # Stem Configuration for Asymmetric Grouped Convolution
    # The first layer will be replaced to handle 24 channels via grouped convolutions.
    # Groups = 8 ensures each group sees 3 input channels (24/8=3), matching ImageNet weights.
    STEM_GROUPS = 8
    STEM_OUT_CHANNELS = 32  # Standard out_channels for EfficientNet-B0 stem

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32
    NUM_EPOCHS = 15  # Sufficient for convergence with pre-trained weights

    # Optimizer settings (Conservative optimization)
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Augmentation Settings
    ROTATION_DEGREES = 15  # +/- 15 degrees

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5
