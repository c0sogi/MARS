import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 500  # Number of samples to use in debug mode

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for caching processed data (idea_19 as requested)
    WORKING_DIR = "./working/idea_19"
    SUBMISSION_DIR = "./submission"

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Paths (Parquet or NPY)
    CACHE_DIR = WORKING_DIR

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Hyperparameters
    # =========================================================================
    IMAGE_SIZE = 32
    NUM_CLASSES = 1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    N_FOLDS = 5
    EPOCHS = 30
    BATCH_SIZE = 64

    # Optimizer
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # Standard for AdamW

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Regularization: Mixup
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2

    # Stochastic Weight Averaging (SWA)
    USE_SWA = True
    SWA_START_EPOCH = 25
    SWA_LR = 1e-4

    # Loss Weights
    # Lambda for the auxiliary MSE loss on log(file_size)
    QUALITY_LOSS_WEIGHT = 0.5

    # =========================================================================
    # Model Definitions
    # =========================================================================
    # Names for the three specialists
    MODEL_STRUCTURAL = "CactusRepVGG_L"  # Input: L channel + Laplacian
    MODEL_CHROMATIC = "CactusResNet_AB"  # Input: A + B channels
    MODEL_HOLISTIC = "CactusNeXt_RGB"  # Input: RGB

    # =========================================================================
    # Compute
    # =========================================================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def print_config():
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        for attr in dir(Config):
            if not attr.startswith("__") and not callable(getattr(Config, attr)):
                print(f"{attr}: {getattr(Config, attr)}")
        print("=" * 30)
