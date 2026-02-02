import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # Compute Environment
    NUM_WORKERS = 8  # Adjust based on available vCPUs (12 available)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for Idea 6
    WORKING_DIR = "./working/idea_6"

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata File Paths (Pre-generated splits)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data & Augmentation
    # -------------------------------------------------------------------------
    NUM_FOLDS = 5
    NUM_CLASSES = 1  # Regression output (continuous scalar for 0-4 scale)

    # Preprocessing & Augmentation
    CLAHE_PROB = 0.5  # Stochastic CLAHE probability
    AUG_ROTATION = 30  # Max rotation in degrees
    AUG_FLIP_PROB = 0.5  # Horizontal/Vertical flip probability

    # -------------------------------------------------------------------------
    # Model Architecture (Hybrid Ensemble)
    # -------------------------------------------------------------------------
    # Global Pooling Settings
    GEM_P = 3.0  # Power for Generalized Mean Pooling
    USE_FP32_GEM = True  # Force GeM to run in float32 to prevent NaN/Overflow

    # Stream 1: CNN Backbone (EfficientNet-B5)
    MODEL_CNN = {
        "name": "tf_efficientnet_b5_ns",
        "img_size": 512,  # High resolution for texture details
        "batch_size": 16,  # Tuned for A100 40GB
        "dropout": 0.2,
        "checkpoint_prefix": "effnet_b5",
    }

    # Stream 2: Transformer Backbone (Swin V2 Base)
    MODEL_TRANS = {
        "name": "swinv2_base_window12_192_22k",  # Will be resized to 384
        "img_size": 384,  # Balanced resolution for global context
        "batch_size": 16,
        "dropout": 0.0,  # Swin handles dropout internally
        "checkpoint_prefix": "swinv2_base",
    }

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    EPOCHS = 12
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    MAX_GRAD_NORM = 10.0

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Stochastic Weight Averaging (SWA)
    USE_SWA = True
    SWA_START_EPOCH = 9  # Start SWA in the final ~25% of training
    SWA_LR = 1e-5  # Constant LR for SWA phase

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    @classmethod
    def setup(cls):
        """Ensures the output directory exists."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)

    @classmethod
    def override(cls, **kwargs):
        """Allows flexible overriding of config parameters."""
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
