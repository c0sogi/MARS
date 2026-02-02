import os
import torch


class Config:
    """
    Central configuration for the Cactus Identification task using
    Custom Wide ResNet with Hybrid Texture-Semantic Aggregation.
    """

    # ==========================================
    # Filesystem Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Pre-generated metadata paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output directories
    WORKING_DIR = "./working/idea_17"
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Compute & Reproducibility
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers
    SEEDS = [0, 1, 2, 3, 4]  # Seeds for Homogeneous Seed Averaging

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    IMAGE_SIZE = (32, 32)
    INPUT_CHANNELS = 3
    NUM_CLASSES = 1

    # Augmentation Strategy: Light Augmentation
    # Only RandomHorizontalFlip and RandomVerticalFlip are permitted
    AUG_HFLIP_PROB = 0.5
    AUG_VFLIP_PROB = 0.5

    # ==========================================
    # Model Architecture: Wide ResNet + Hybrid Head
    # ==========================================
    # Backbone: Wide ResNet (3 Stages)
    # Channels scale: [32, 64, 128] to ensure raw capacity
    BACKBONE_CHANNELS = [32, 64, 128]
    USE_SE_BLOCKS = False  # Excluded to minimize latency
    KERNEL_SIZE = 3  # 3x3 convolutions exclusively

    # Hybrid Head Configuration
    # Stream 1 (Texture): Stage 2 (16x16, 64ch) -> Global Covariance Pooling
    # Stream 2 (Semantic): Stage 3 (8x8, 128ch) -> Global Average Pooling
    TEXTURE_STREAM_IDX = 1  # Index of the stage output for texture stream
    SEMANTIC_STREAM_IDX = 2  # Index of the stage output for semantic stream

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # Standard for AdamW

    # Learning Rate Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Optimization
    EARLY_STOPPING_PATIENCE = 7  # Stop if val_auc doesn't improve for 7 epochs

    # ==========================================
    # Inference / TTA
    # ==========================================
    # Test Time Augmentation: Original + HFlip + VFlip
    USE_TTA = True

    # ==========================================
    # Debugging
    # ==========================================
    # Set DEBUG to True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200
