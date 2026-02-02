import os
import torch


class Config:
    """
    Global configuration for the Bird Species Classification pipeline.
    Implements the 'Optimized Strategy' with Heterogeneous Ensemble and SWA.
    """

    # ==========================================
    # System & Paths
    # ==========================================
    PROJECT_NAME = "idea_17"
    INPUT_ROOT = "./input"
    OUTPUT_ROOT = "./working"
    SUBMISSION_ROOT = "./submission"

    # Metadata Paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Data Source Paths
    # Strictly using standard spectrograms as per strategy
    SPECTROGRAM_DIR = os.path.join(INPUT_ROOT, "supplemental_data", "spectrograms")

    # Working Directories
    IDEA_DIR = os.path.join(OUTPUT_ROOT, PROJECT_NAME)
    CACHE_DIR = os.path.join(IDEA_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(IDEA_DIR, "checkpoints")

    # Submission
    SUBMISSION_FILE = os.path.join(SUBMISSION_ROOT, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = (224, 224)
    IN_CHANNELS = 3  # Replicating single channel to 3 for pre-trained weights
    NUM_CLASSES = 19
    NUM_FOLDS = 5

    # Augmentation Strategy
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4
    ENABLE_TIME_SHIFT = True  # Horizontal translation with zero padding
    ENABLE_PHOTOMETRIC = True  # Brightness and Contrast jitter
    ENABLE_HORIZONTAL_FLIP = False  # Strictly disabled

    # ==========================================
    # Model Configuration
    # ==========================================
    # Heterogeneous Ensemble Backbones
    MODEL_BACKBONES = ["resnet18", "efficientnet_b0", "densenet121"]
    PRETRAINED = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Optimization
    EPOCHS = 50
    # Effective Batch Size = 64
    # Achieved via Gradient Accumulation: 32 * 2 = 64
    PHYSICAL_BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 2

    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # High weight decay for regularization

    # Scheduler & SWA
    # Strategy: Cosine Annealing -> Constant SWA LR
    USE_SWA = True
    SWA_START_EPOCH_PCT = 0.75  # Start SWA at 75% of training
    SWA_LR = 1e-4  # Constant low LR for SWA phase

    # ==========================================
    # Debugging / Development
    # ==========================================
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50  # Number of samples to use if DEBUG is True

    def __init__(self, debug=False, epochs=None):
        """
        Initialize configuration with optional overrides.

        Args:
            debug (bool): If True, enables debug mode (subset of data, fewer epochs).
            epochs (int, optional): Override the number of training epochs.
        """
        # Create necessary directories
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_ROOT, exist_ok=True)

        if debug:
            self.DEBUG = True
            self.EPOCHS = 5 if epochs is None else epochs
        elif epochs is not None:
            self.EPOCHS = epochs

    @property
    def SWA_START_EPOCH(self):
        """Calculate the epoch to start SWA based on percentage."""
        return int(self.EPOCHS * self.SWA_START_EPOCH_PCT)
