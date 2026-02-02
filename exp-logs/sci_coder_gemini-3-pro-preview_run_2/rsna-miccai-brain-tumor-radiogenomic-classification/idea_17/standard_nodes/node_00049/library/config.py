import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Reproducibility & Hardware
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available, leaving some overhead
    NUM_WORKERS = 8

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data (idea_17 specific)
    WORKING_DIR = "./working/idea_17"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # --------------------------------------------------------------------------
    IMG_SIZE = 224

    # Stacking Logic
    NUM_SLICES = 3  # Center slice + neighbors
    STRIDE = 5  # Fixed stride for neighbors

    # Input Dimensions
    # 4 Modalities (FLAIR, T1w, T1wCE, T2w) * 3 Slices each = 12 Channels
    IN_CHANNELS = 12

    # Depth Filtering (15% - 85% of volume)
    DEPTH_MIN = 0.15
    DEPTH_MAX = 0.85

    # Dual-Anchor Strategy Configuration
    # View 1: Anatomical Anchor (FLAIR Integral)
    ANCHOR_1_MODALITY = "FLAIR"
    ANCHOR_1_METHOD = "sum"

    # View 2: Pathological Anchor (T1wCE Greedy Max)
    ANCHOR_2_MODALITY = "T1wCE"
    ANCHOR_2_METHOD = "max"

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # --------------------------------------------------------------------------
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = 1  # Binary classification (BCEWithLogitsLoss)
    STEM_GROUPS = 4  # Grouped convolution groups (one per modality)
    DROP_RATE = 0.3  # Dropout rate for the classification head

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32  # Fits comfortably in A100 40GB with EfficientNet-B0
    EPOCHS = 20

    # Optimization
    LEARNING_RATE = 1e-4  # Low LR to preserve pre-trained features
    WEIGHT_DECAY = 1e-2  # Aggressive weight decay

    # Early Stopping
    PATIENCE = 5

    # Augmentation
    ROTATION_DEGREES = 15
