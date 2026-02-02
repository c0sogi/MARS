import os
import torch


class Config:
    """
    Central configuration for the Ultra-Wide SE-RepNeXt Cactus Classifier.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working directory for intermediate artifacts (checkpoints, cache)
    WORKING_DIR = "./working/idea_37"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Parameters
    # =========================================================================
    IMAGE_SIZE = (32, 32)
    NUM_CLASSES = 1

    # Augmentation Strategy: Light augmentation only
    AUG_HORIZONTAL_FLIP = True
    AUG_VERTICAL_FLIP = True
    AUG_ROTATION = False  # Explicitly disabled
    AUG_COLOR_JITTER = False  # Explicitly disabled

    # =========================================================================
    # Model Architecture: Ultra-Wide SE-RepNeXt with Spatial Fusion
    # =========================================================================
    MODEL_NAME = "UltraWideSERepNeXt"

    # "Ultra-Wide" Channel Configuration as per strategy
    # Stage 1 -> Stage 2 -> Stage 3
    BACKBONE_CHANNELS = [96, 192, 384]

    # RepNeXt Grouped Convolution Cardinality
    CARDINALITY = 32

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Homogeneous Seed Averaging
    SEEDS = [0, 1, 2, 3, 4]

    # Optimization
    EPOCHS = 20
    BATCH_SIZE = 128  # High batch size for stability
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MODE = "max"  # Monitor AUC (maximize)

    # Debugging / Prototyping
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 1000  # Number of samples to use in debug mode

    # =========================================================================
    # Compute & System
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True
