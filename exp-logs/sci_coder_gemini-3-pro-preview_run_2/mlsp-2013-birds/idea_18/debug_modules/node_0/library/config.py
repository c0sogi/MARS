import os
import torch


class Config:
    """
    Configuration for Bird Species Classification Task.
    Implements the 'Heterogeneous Ensemble with Selective Signal-Noise Mixup' strategy.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_18"

    # Metadata Files
    # Note: The strategy involves 5-fold CV on the development set.
    # The provided metadata splits (train.csv, val.csv) are a single 80/20 split of Fold 0.
    # The training script will need to combine these or use the raw CVfolds_2.txt if implementing full 5-fold.
    # For simplicity, we point to the generated metadata files.
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission
    SUBMISSION_PATH = "./submission/submission.csv"

    # Data Source Override
    # Strategy requires "Filtered Spectrograms". Metadata points to "spectrograms".
    # The dataloader should replace 'spectrograms' with this folder name in paths.
    SPECTROGRAM_DIR_NAME = "filtered_spectrograms"

    # =========================================================================
    # Data & Compute
    # =========================================================================
    SEED = 42
    NUM_CLASSES = 19
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging / Development
    DEBUG = False
    DEBUG_DATA_SUBSET = 0.1  # Fraction of data to use if DEBUG is True

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Tri-Backbone Heterogeneous Ensemble
    BACKBONES = ["resnet18", "efficientnet_b0", "densenet121"]

    # Multi-Sample Dropout Head
    NUM_DROPOUT_SAMPLES = 5
    DROPOUT_RATE = 0.2

    # =========================================================================
    # Input Topology (Multi-Resolution Strategy)
    # =========================================================================
    # Format: (Freq/Height, Time/Width)
    IMG_SIZE_DEFAULT = (224, 448)  # For ResNet18 and EfficientNet-B0
    IMG_SIZE_DENSE = (160, 320)  # For DenseNet121

    @staticmethod
    def get_image_size(backbone_name):
        """Returns the specific input resolution for a given backbone."""
        if "densenet" in backbone_name:
            return Config.IMG_SIZE_DENSE
        return Config.IMG_SIZE_DEFAULT

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    NUM_FOLDS = 5
    BATCH_SIZE = 32

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Schedule
    # Strategy: "Target 1000 Total Update Steps per fold"
    MAX_STEPS = 1000

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10  # Epochs (approximate, depending on eval freq)

    # =========================================================================
    # Augmentation Strategy
    # =========================================================================
    # Selective Signal-Noise Mixup
    USE_SELECTIVE_MIXUP = True
    MIXUP_ALPHA = 1.0

    # Test-Time Augmentation
    TTA_STEPS = 4  # Original + 3 Time-Roll shifts

    @classmethod
    def setup(cls):
        """Ensures necessary working directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.SUBMISSION_PATH), exist_ok=True)

    def __init__(self, **kwargs):
        """
        Allows initialization with overrides for flexibility.
        Example: config = Config(BATCH_SIZE=16, DEBUG=True)
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
