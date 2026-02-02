import os
import torch


class Config:
    # ==== General Configuration ====
    SEED = 42
    DEBUG = False  # Set to True for fast debugging runs
    EXP_NAME = "idea_15"

    # ==== Directories ====
    # Input data (Read-Only)
    INPUT_DIR = "./input"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")
    METADATA_DIR = "./metadata"

    # Working directory for caching and checkpoints
    WORKING_DIR = os.path.join("./working", EXP_NAME)

    # Output directory for submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==== Data Configuration ====
    NUM_CLASSES = 4
    CLASS_LABELS = ["healthy", "multiple_diseases", "rust", "scab"]

    # Image resolution: 384x384 to preserve small lesion details
    IMG_SIZE = 384

    # Batch size: Adjusted for A100 40GB with 384x384 resolution
    BATCH_SIZE = 16
    NUM_WORKERS = 4

    # ==== Model Configuration ====
    # Dual-Backbone Heterogeneous Ensemble
    # 1. EfficientNetV2-M: Texture expert (Fused-MBConv)
    # 2. Swin-Small: Context expert (Long-range dependencies)
    # Using specific timm tags for pretrained weights
    BACKBONES = [
        "tf_efficientnetv2_m",
        "swin_small_patch4_window7_224",
    ]

    # Pooling strategy: Multi-Level Generalized Mean Pooling
    POOLING_TYPE = "multi_level_gem"
    GEM_P = 3.0  # Initial power for GeM

    # Model EMA (Exponential Moving Average)
    USE_EMA = True
    EMA_DECAY = 0.999

    # ==== Training Configuration ====
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Optimizer & Scheduler
    OPTIMIZER = "AdamW"
    SCHEDULER = "CosineAnnealingLR"
    MIN_LR = 1e-6

    # Early Stopping
    PATIENCE = 10  # Relaxed patience to allow EMA convergence

    # Loss Function
    USE_CLASS_WEIGHTS = True  # Inverse frequency weights
    LABEL_SMOOTHING = 0.05

    # Mixed Precision
    USE_AMP = True

    # ==== Augmentation Configuration ====
    # Strategy: Strong Geometric, No Photometric, No Occlusion
    AUG_PROB = 0.5

    # Geometric
    RANDOM_FLIP_PROB = 0.5
    SHIFT_SCALE_ROTATE_PROB = 0.5
    SHIFT_LIMIT = 0.1
    SCALE_LIMIT = 0.2
    ROTATE_LIMIT = 15

    # Exclusions (Explicitly False)
    USE_CUTOUT = False
    USE_MIXUP = False
    USE_CUTMIX = False
    USE_COLOR_JITTER = False

    # ==== Inference Configuration ====
    # Test-Time Augmentation (TTA)
    # Domain-Aware: Horizontal Flip only (Vertical flip violates gravity priors)
    TTA_FLIP_HORIZONTAL = True
    TTA_FLIP_VERTICAL = False

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_cache_path(cls, filename):
        """Helper to get full path for cached files in the working directory."""
        return os.path.join(cls.WORKING_DIR, filename)
