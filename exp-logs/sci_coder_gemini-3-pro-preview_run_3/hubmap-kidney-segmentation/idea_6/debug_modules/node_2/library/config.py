import os
import torch


class Config:
    """
    Configuration class for the HuBMAP FTU Detection Pipeline.
    Centralizes all hyperparameters, paths, and model settings.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for artifacts (checkpoints, logs, cache)
    WORKING_DIR = "./working/idea_6"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Model Checkpoint Path
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Architecture
    ARCH = "UnetPlusPlus"
    BACKBONE = "tf_efficientnetv2_m"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 3
    NUM_CLASSES = 1

    # Input Resolution
    # Increased to 768x768 based on EfficientNetV2 efficiency
    TILE_SIZE = 768

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Hardware settings (A100 40GB)
    # Batch size 4 is safe for 768x768 with V2-M and Deep Supervision
    BATCH_SIZE = 4
    NUM_WORKERS = 4  # 12 vCPUs available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    EPOCHS = 15
    LR = 1e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS

    # Loss Weights for Deep Supervision
    # Weights for outputs at scales [1.0, 0.5, 0.25, 0.125]
    LOSS_WEIGHTS = [1.0, 0.5, 0.25, 0.125]

    # =========================================================================
    # Data & Augmentation
    # =========================================================================
    # Random Seed for Reproducibility
    SEED = 42

    # Augmentation Parameters
    AUG_PROB = 0.5
    ROTATE_LIMIT = 90
    SCALE_LIMIT = 0.2
    SHIFT_LIMIT = 0.1

    # Inference
    THRESHOLD = 0.5

    # =========================================================================
    # Debugging / Development
    # =========================================================================
    # Set DEBUG to True to run on a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print("Configuration:")
        print("=" * 30)
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key}: {value}")
        print("=" * 30)
