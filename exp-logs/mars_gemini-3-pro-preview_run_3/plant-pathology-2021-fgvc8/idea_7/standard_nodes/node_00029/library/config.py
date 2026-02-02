import os
import torch


class Config:
    # ==========================================
    # Experiment Identification
    # ==========================================
    EXPERIMENT_NAME = "idea_7"
    SEED = 42

    # ==========================================
    # Data Configuration
    # ==========================================
    # Input paths
    INPUT_DIR = "./input"
    TRAIN_METADATA = "./metadata/train.csv"
    VAL_METADATA = "./metadata/val.csv"
    TEST_METADATA = "./metadata/test.csv"

    # Output paths
    WORKING_DIR = f"./working/{EXPERIMENT_NAME}"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    CACHE_DIR = WORKING_DIR  # For parquet caches

    # Image parameters
    IMAGE_SIZE = 384
    NUM_CLASSES = 6
    # Labels based on data analysis sorted alphabetically
    LABELS = [
        "complex",
        "frog_eye_leaf_spot",
        "healthy",
        "powdery_mildew",
        "rust",
        "scab",
    ]

    # ==========================================
    # Model Configuration
    # ==========================================
    MODEL_NAME = "convnext_small"  # Will use timm
    POOLING = "gem"  # Generalized Mean Pooling
    PRETRAINED = True
    DROP_PATH_RATE = 0.2  # Stochastic depth rate

    # ==========================================
    # Training Configuration
    # ==========================================
    EPOCHS = 50
    BATCH_SIZE = 32  # Adjusted for 384x384 on A100 (40GB)
    NUM_WORKERS = 12

    # Optimizer
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.05
    MIN_LR = 1e-6

    # Scheduler
    WARMUP_EPOCHS = 5

    # EMA (Exponential Moving Average)
    USE_EMA = True
    EMA_DECAY = 0.999  # Tuned for 50 epochs

    # ==========================================
    # Regularization & Augmentation
    # ==========================================
    # MixUp / CutMix
    MIXUP_ALPHA = 0.8
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 1.0  # Probability of applying either MixUp or CutMix
    SWITCH_PROB = 0.5  # Probability of switching between MixUp and CutMix

    # Geometric Augmentations
    RANDOM_RESIZE_CROP_SCALE = (0.5, 1.0)  # Aggressive minimum scale
    FLIP_PROB = 0.5  # For both horizontal and vertical

    # ==========================================
    # Inference Configuration
    # ==========================================
    USE_TTA = True  # Test Time Augmentation (Original + HFlip + VFlip)
    TTA_FLIPS = [[2], [3]]  # Horizontal and Vertical flip dims for torch
    THRESHOLD = 0.5

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup_directories(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)


# Initialize directories on import
Config.setup_directories()
