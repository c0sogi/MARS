import os
import torch


class Config:
    """
    Configuration for the Dynamic Deep Supervision Stratified Ensemble strategy.

    Strategy Overview:
    - Architecture: U-Net++ with ResNeXt-50 (32x4d) encoder.
    - Input: 3-channel (Seismic, Seismic, Depth), padded to 128x128.
    - Training: 5-Fold Stratified CV, 80 Epochs total.
    - Curriculum:
        - Phase 1 (0-20): Deep Supervision (L1-L4), BCE+Dice, LR 5e-4.
        - Phase 2 (21-80): Final Head Only, Lovasz-Hinge, LR 5e-5.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea/experiment
    WORK_DIR = "./working/idea_15"
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORK_DIR, "submission")
    CACHE_DIR = os.path.join(WORK_DIR, "cache")

    # Ensure working directories exist
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    # Original image dimensions
    ORIG_HEIGHT = 101
    ORIG_WIDTH = 101

    # Training/Inference dimensions (padded)
    IMG_HEIGHT = 128
    IMG_WIDTH = 128

    # Input Channels: Seismic, Seismic, Depth
    IN_CHANNELS = 3

    # Metric Thresholds (0.5 to 0.95 step 0.05)
    IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    ENCODER_NAME = "resnext50_32x4d"
    ENCODER_WEIGHTS = "imagenet"

    # U-Net++ Decoder settings
    # Lightweight channels to prevent overfitting
    DECODER_CHANNELS = (256, 128, 64, 32, 16)

    # Attention mechanism
    DECODER_ATTENTION = "scse"

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    FOLDS = 5
    BATCH_SIZE = 64
    NUM_WORKERS = 4  # 12 vCPUs available, 4 is usually safe/optimal for DataLoader

    # Total runtime budget management
    TOTAL_EPOCHS = 80

    # Phase 1: Structural Warm-up (Deep Supervision Active)
    PHASE1_EPOCHS = 20
    PHASE1_LR = 5e-4

    # Phase 2: Metric Fine-tuning (Deep Supervision Disabled, Lovasz Loss)
    # Must be conservative (<= 1e-4)
    PHASE2_LR = 5e-5

    # Optimization
    WEIGHT_DECAY = 1e-4

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MIN_LR = 1e-6

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Debugging
    # =========================================================================
    # Set to a small integer (e.g., 100) to run on a subset of data for testing
    # Set to None for full training
    DEBUG_SAMPLES = None
