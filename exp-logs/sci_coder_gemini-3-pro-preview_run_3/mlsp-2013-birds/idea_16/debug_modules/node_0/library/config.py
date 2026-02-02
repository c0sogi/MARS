import os


class Config:
    # =========================================================================
    # General Configuration
    # =========================================================================
    PROJECT_NAME = "idea_16"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    NUM_WORKERS = 4  # Number of data loading workers

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Source images: Standard spectrograms (Not filtered)
    SPECTROGRAM_DIR = os.path.join(INPUT_ROOT, "supplemental_data", "spectrograms")

    # Output directories
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Metadata files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    IMAGE_SIZE = (224, 224)  # Resize to standard input size
    NUM_CLASSES = 19
    CHANNELS = 3  # Convert 1-channel spectrogram to 3-channel pseudo-RGB

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    N_FOLDS = 5
    EPOCHS = 40
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # High weight decay for regularization

    # Scheduler settings (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # =========================================================================
    # Model Configuration
    # =========================================================================
    # Heterogeneous Ensemble Backbones
    BACKBONES = ["resnet18", "efficientnet_b0", "densenet121"]

    # Multi-Sample Dropout (MSD) Settings
    MSD_DROPOUT_RATE = 0.5
    MSD_NUM_SAMPLES = 8

    # Top-K Checkpoint Averaging
    TOP_K_CHECKPOINTS = 3  # Number of best checkpoints to average per fold

    @classmethod
    def setup(cls):
        """Creates the necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
