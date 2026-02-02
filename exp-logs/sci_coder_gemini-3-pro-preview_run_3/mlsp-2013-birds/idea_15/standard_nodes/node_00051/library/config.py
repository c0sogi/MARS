import os
import torch


class Config:
    # --------------------
    # General Configuration
    # --------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SUBSET_SIZE = 50
    NUM_WORKERS = 4  # Optimized for the 12 vCPUs available

    # --------------------
    # Data Configuration
    # --------------------
    IMG_SIZE = 224  # Strictly 224x224 as per strategy
    NUM_CLASSES = 19
    N_FOLDS = 5

    # Augmentation
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4

    # --------------------
    # Path Configuration
    # --------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"

    # Source Data
    # Strictly using standard spectrograms, avoiding filtered ones
    SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------
    # Model Configuration
    # --------------------
    # Heterogeneous Ensemble Backbones
    BACKBONES = ["resnet18", "efficientnet_b0", "densenet121"]
    POOLING_TYPE = "concat"  # Concatenated Pooling (GAP + GMP)
    PRETRAINED = True
    IN_CHANNELS = 3  # Pseudo-RGB (3-Channel Rule)

    # --------------------
    # Training Configuration
    # --------------------
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 20  # Relaxed patience for robust convergence

    # Scheduler
    SCHEDULER_T_MAX = EPOCHS
    SCHEDULER_MIN_LR = 1e-6

    # --------------------
    # EMA Configuration
    # --------------------
    USE_EMA = True
    EMA_DECAY = 0.95  # Low decay rate for small dataset/short training

    # --------------------
    # Compute
    # --------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        """Creates necessary output directories and prints configuration."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        print(f"Config Setup Complete:")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Image Size: {cls.IMG_SIZE}")
        print(f"  EMA Decay: {cls.EMA_DECAY}")
        print(f"  Backbones: {cls.BACKBONES}")
        print(f"  Working Dir: {cls.WORKING_DIR}")
