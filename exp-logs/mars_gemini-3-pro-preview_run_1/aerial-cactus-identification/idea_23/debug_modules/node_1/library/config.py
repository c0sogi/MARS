import os
import torch


class Config:
    # ==========================================
    # Experiment Control
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    EXPERIMENT_NAME = "idea_23"

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    # Image directories
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata paths (pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working directory for artifacts
    WORK_DIR = os.path.join("./working", EXPERIMENT_NAME)
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary writeable directories exist
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Constants
    # ==========================================
    IMG_SIZE = 32
    IN_CHANNELS = 3
    NUM_CLASSES = 1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    N_FOLDS = 5
    EPOCHS = 30
    BATCH_SIZE = 128
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # Optimizer (AdamW)
    LR = 1e-3
    WEIGHT_DECAY = 1e-2
    MIN_LR = 1e-6

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS  # Cycle length

    # ==========================================
    # Strategy Specifics
    # ==========================================
    # Mixup Regularization
    MIXUP_ALPHA = 0.2

    # Auxiliary Quality Supervision
    # Weight for the file-size regression loss (MSE)
    AUX_LOSS_WEIGHT = 0.1

    # Stochastic Weight Averaging (SWA)
    USE_SWA = True
    SWA_START_EPOCH = 20
    SWA_LR = 1e-4

    # ==========================================
    # Model Architecture (Custom RepVGG)
    # ==========================================
    # Width multiplier for the backbone
    WIDTH_MULTIPLIER = 1.0
    # Convolution groups
    GROUPS = 1

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Logging
    # ==========================================
    PRINT_FREQ = 10  # Print metrics every N batches
