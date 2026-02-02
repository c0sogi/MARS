import os
import torch


class Config:
    """
    Configuration class for Salt Segmentation Task.
    Implements the 'Adaptive High-Fidelity SWA Ensemble' strategy settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    PROJECT_NAME = "SaltSegmentation_Idea17"
    DEBUG = False  # Set to True for quick debugging runs
    DEBUG_SIZE = 50  # Number of samples to use in debug mode

    # =========================================================================
    # Directory Paths
    # =========================================================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Task-specific cache directory (Idea 17)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_17")
    CHECKPOINT_DIR = os.path.join(CACHE_DIR, "checkpoints")
    LOG_DIR = os.path.join(CACHE_DIR, "logs")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files
    # Note: We will merge train and val metadata for 5-fold cross-validation
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Create necessary directories
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data & Image Processing
    # =========================================================================
    # Original image dimensions
    ORIG_HEIGHT = 101
    ORIG_WIDTH = 101

    # Model input dimensions (Padded/Resized)
    # Using 128x128 for U-Net compatibility (divisible by 32)
    MODEL_HEIGHT = 128
    MODEL_WIDTH = 128

    # Input Channels: 3 (Seismic, Seismic, Depth)
    IN_CHANNELS = 3

    # Normalization
    # We do NOT use ImageNet normalization as we are using custom depth channels
    # Images are scaled to [0, 1]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Architecture: U-Net++ (Nested U-Net)
    ARCH = "UnetPlusPlus"

    # Encoder: SE-ResNeXt-50 (32x4d)
    ENCODER_NAME = "se_resnext50_32x4d"
    ENCODER_WEIGHTS = "imagenet"

    # Decoder
    # Lightweight channels to prevent overfitting
    DECODER_CHANNELS = (256, 128, 64, 32, 16)

    # Attention
    # Concurrent Spatial and Channel Squeeze & Excitation
    DECODER_ATTENTION_TYPE = "scse"

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    NUM_FOLDS = 5
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Total training budget
    TOTAL_EPOCHS = 80

    # --- Phase 1: Structural Warmup ---
    # Loss: BCE + Dice
    # Deep Supervision: Active
    LR_PHASE1 = 5e-4
    PHASE1_PATIENCE = 5  # Epochs with no improvement to trigger switch

    # --- Phase 2: Metric Fine-Tuning ---
    # Loss: Lovasz-Hinge
    # Deep Supervision: Disabled (Final head only)
    LR_PHASE2 = 5e-5

    # --- Stochastic Weight Averaging (SWA) ---
    # Start SWA in the last 20% of training
    SWA_START_EPOCH_RATIO = 0.8
    SWA_LR = 1e-5

    # =========================================================================
    # Evaluation & Metrics
    # =========================================================================
    # IoU Thresholds for mAP calculation
    IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # TTA
    USE_TTA = True  # Horizontal Flip TTA
