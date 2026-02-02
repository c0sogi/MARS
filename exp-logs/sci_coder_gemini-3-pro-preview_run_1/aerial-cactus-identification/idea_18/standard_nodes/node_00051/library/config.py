import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # =========================================================================
    # Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Main working directory for this specific idea/iteration
    WORKING_DIR = "./working/idea_18"

    # Sub-directories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Files (Numpy format)
    # We cache images (uint8 or float), labels, file sizes, and IDs
    CACHE_FILE_MAP = {
        "train_imgs": os.path.join(CACHE_DIR, "train_imgs.npy"),
        "train_labels": os.path.join(CACHE_DIR, "train_labels.npy"),
        "train_fsizes": os.path.join(CACHE_DIR, "train_fsizes.npy"),
        "train_ids": os.path.join(CACHE_DIR, "train_ids.npy"),
        "test_imgs": os.path.join(CACHE_DIR, "test_imgs.npy"),
        "test_ids": os.path.join(CACHE_DIR, "test_ids.npy"),
        "test_fsizes": os.path.join(CACHE_DIR, "test_fsizes.npy"),
        "val_imgs": os.path.join(CACHE_DIR, "val_imgs.npy"),
        "val_labels": os.path.join(CACHE_DIR, "val_labels.npy"),
        "val_fsizes": os.path.join(CACHE_DIR, "val_fsizes.npy"),
        "val_ids": os.path.join(CACHE_DIR, "val_ids.npy"),
    }

    # =========================================================================
    # Data Parameters
    # =========================================================================
    IMG_SIZE = 32
    NUM_CLASSES = 1

    # Normalization constants (calculated from dataset analysis)
    # RGB Mean and Std
    NORM_MEAN = [0.5034, 0.4520, 0.4683]  # R, G, B
    NORM_STD = [0.1514, 0.1399, 0.1535]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    NUM_FOLDS = 5
    BATCH_SIZE = 128
    EPOCHS = 30

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Mixup
    MIXUP_ALPHA = 0.2

    # Stochastic Weight Averaging (SWA)
    USE_SWA = True
    SWA_START_EPOCH = 25
    SWA_LR = 1e-4

    # Multi-Task Learning Weight (Regression loss for file size)
    MTL_LOSS_WEIGHT = 0.1

    # =========================================================================
    # Model Specifications (Heterogeneous Dual-Domain Stacking)
    # =========================================================================
    # Defines the ensemble members.
    # 'in_chans': 3 for Spatial (RGB), 4 for Texture (RGB + Laplacian)
    MODEL_CONFIGS = {
        "RepVGG_Spatial": {
            "arch": "RepVGG",
            "in_chans": 3,
            "use_film": True,
            "use_mtl": True,
        },
        "RepVGG_Texture": {
            "arch": "RepVGG",
            "in_chans": 4,  # RGB + Laplacian Edge Map
            "use_film": True,
            "use_mtl": True,
        },
        "ResNet_Spatial": {
            "arch": "ResNet",
            "in_chans": 3,
            "use_film": True,
            "use_mtl": True,
        },
    }

    @classmethod
    def setup_directories(cls):
        """Creates necessary directories if they don't exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories initialized at {cls.WORKING_DIR}")

    @classmethod
    def get_cache_path(cls, key):
        """Returns the path for a specific cache key."""
        return cls.CACHE_FILE_MAP.get(key)
