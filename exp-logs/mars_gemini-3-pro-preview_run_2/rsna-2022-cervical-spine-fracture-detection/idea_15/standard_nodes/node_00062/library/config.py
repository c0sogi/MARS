import os
import torch


class Config:
    """
    Configuration for the Calibrated 2.5D Multi-Scale Sequence Network.
    This module defines all hyperparameters, file paths, and constants
    used throughout the training and inference pipeline.
    """

    # =========================================================================
    # 1. Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata paths (pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working directory for model checkpoints, cache, and submission
    WORKING_DIR = "./working/idea_15"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # =========================================================================
    # 2. Data Configuration
    # =========================================================================
    # Input image resolution (H, W).
    # 256x256 is a balance between preserving fracture details and sequence memory usage.
    IMAGE_SIZE = (256, 256)

    # Sequence length (Z-axis depth)
    # 96 slices ensures high-density sampling to capture thin fractures.
    SEQ_LEN = 96

    # Input channels: 3 for 2.5D stacking (Slice z-1, Slice z, Slice z+1)
    IN_CHANNELS = 3

    # DataLoader settings
    NUM_WORKERS = 4

    # =========================================================================
    # 3. Model Configuration
    # =========================================================================
    # Backbone for feature extraction (timm library name)
    BACKBONE_NAME = "tf_efficientnet_b4_ns"

    # Sequence modeling (Bi-LSTM) settings
    LSTM_HIDDEN_SIZE = 256
    LSTM_LAYERS = 2
    DROPOUT = 0.2

    # Output Classes: 7 Vertebrae (C1-C7) + 1 Patient Overall
    NUM_CLASSES = 8
    TARGET_COLUMNS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

    # =========================================================================
    # 4. Training Hyperparameters
    # =========================================================================
    # Random Seed for full reproducibility
    SEED = 42

    # Debug Flag: If True, drastically reduces dataset size for pipeline verification
    DEBUG = False

    # Training Duration
    EPOCHS = 10

    # Batch Size Configuration
    # Physical batch size per GPU step (limited by VRAM for B4 + 96 seq len)
    BATCH_SIZE = 1

    # Gradient Accumulation Steps
    # Effective Batch Size = BATCH_SIZE * ACCUMULATION_STEPS = 16
    ACCUMULATION_STEPS = 16

    # Optimizer Settings (AdamW)
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    MAX_GRAD_NORM = 10.0

    # Learning Rate Scheduler (CosineAnnealingLR)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping to prevent overfitting
    EARLY_STOPPING_PATIENCE = 3

    # =========================================================================
    # 5. Loss & Metric Configuration
    # =========================================================================
    # Weighted Multi-Label Logarithmic Loss Weights
    # The 'patient_overall' label is weighted higher (7.0) than individual vertebrae (1.0)
    # to align with the competition metric which prioritizes the overall diagnosis.
    # Order corresponds to TARGET_COLUMNS.
    LOSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0]

    # Note: Positive class weighting is explicitly set to 1.0 (disabled) in the
    # trainer to ensure probabilistic calibration.

    # =========================================================================
    # 6. Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
