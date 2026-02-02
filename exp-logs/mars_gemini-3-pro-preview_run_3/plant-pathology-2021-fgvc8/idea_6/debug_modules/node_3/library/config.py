import os
import torch


class Config:
    # ==========================
    # General Settings
    # ==========================
    PROJECT_NAME = "apple_disease_detection"
    IDEA_NAME = "idea_6"
    SEED = 42
    DEBUG = False  # Toggle for debugging on small subsets

    # ==========================
    # Directories & Paths
    # ==========================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Output Paths
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (for processed data frames/features)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_cache.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_cache.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_cache.parquet")

    # ==========================
    # Data Configuration
    # ==========================
    IMG_SIZE = 384
    NUM_WORKERS = 12

    # Class Labels (Alphabetically sorted)
    CLASSES = [
        "complex",
        "frog_eye_leaf_spot",
        "healthy",
        "powdery_mildew",
        "rust",
        "scab",
    ]
    NUM_CLASSES = len(CLASSES)
    LABEL2ID = {label: i for i, label in enumerate(CLASSES)}
    ID2LABEL = {i: label for i, label in enumerate(CLASSES)}

    # ==========================
    # Model Configuration
    # ==========================
    MODEL_NAME = "convnext_small"
    PRETRAINED = True
    DROP_PATH_RATE = 0.2  # Stochastic depth rate

    # Pooling Head
    POOLING = "gem"  # Generalized Mean Pooling
    GEM_P = 3.0  # Initial power for GeM
    GEM_TRAINABLE = True

    # ==========================
    # Training Configuration
    # ==========================
    EPOCHS = 50
    BATCH_SIZE = 32  # Optimized for A100 40GB with 384x384 ConvNeXt-Small

    # Optimizer (AdamW)
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.05
    EPS = 1e-8
    BETAS = (0.9, 0.999)

    # Scheduler (Cosine Annealing)
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 5

    # EMA (Exponential Moving Average)
    USE_EMA = True
    EMA_DECAY = 0.999

    # ==========================
    # Augmentation & Regularization
    # ==========================
    # MixUp / CutMix
    MIXUP_ALPHA = 0.8
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 1.0  # Probability to apply either MixUp or CutMix
    SWITCH_PROB = 0.5  # Probability to switch between MixUp and CutMix

    # Geometric Augmentations
    RANDOM_RESIZE_CROP_SCALE = (0.5, 1.0)
    FLIP_PROB = 0.5  # Probability for Horizontal and Vertical flips

    # ==========================
    # Inference
    # ==========================
    TTA = True  # Test Time Augmentation (Horizontal & Vertical Flips)
    THRESHOLD = 0.5  # Threshold for multi-label binary classification

    # ==========================
    # Hardware
    # ==========================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
