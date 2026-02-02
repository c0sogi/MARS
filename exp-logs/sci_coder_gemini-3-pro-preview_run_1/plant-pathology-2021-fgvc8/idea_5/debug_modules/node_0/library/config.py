import os
import torch


class Config:
    """
    Configuration class for Apple Disease Detection Task.
    Implements the 'Idea 5' strategy: MaxViT-Tiny, 384px, Label Smoothing.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run with a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 100

    # =========================================================================
    # Directory & File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"

    # Ensure working directory exists for caching and outputs
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    LOG_PATH = os.path.join(WORKING_DIR, "train_log.txt")

    # Cache Paths (for deterministic data processing)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Hybrid CNN-Transformer architecture (MaxViT)
    MODEL_NAME = "maxvit_tiny_tf_384.in1k"
    IMG_SIZE = 384
    NUM_CLASSES = 6

    # Class Labels (Sorted Alphabetically for consistency)
    CLASSES = [
        "complex",
        "frog_eye_leaf_spot",
        "healthy",
        "powdery_mildew",
        "rust",
        "scab",
    ]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 15
    BATCH_SIZE = 16  # Adjusted for 384x384 resolution on 40GB GPU
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2  # Standard for AdamW
    LABEL_SMOOTHING = 0.05
    MAX_GRAD_NORM = 10.0

    # Scheduler Settings (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # =========================================================================
    # Compute & Hardware
    # =========================================================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = True  # Automatic Mixed Precision required for MaxViT memory efficiency

    # =========================================================================
    # Inference
    # =========================================================================
    THRESHOLD = 0.5  # Probability threshold for multi-label classification
