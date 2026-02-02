import os


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    # Root directory for this specific experiment
    WORK_DIR = "./working/idea_14"

    # Input directories
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Data source paths
    SPECTROGRAM_DIR = os.path.join(INPUT_ROOT, "supplemental_data", "spectrograms")
    FILTERED_SPECTROGRAM_DIR = os.path.join(
        INPUT_ROOT, "supplemental_data", "filtered_spectrograms"
    )

    # Output directories
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORK_DIR, "submission")

    # Metadata files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    NUM_CLASSES = 19
    N_FOLDS = 5

    # Debugging / Development
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLES = 20  # Number of samples to use in debug mode

    # ==========================================
    # Data Preprocessing
    # ==========================================
    IMG_SIZE = (224, 224)
    # 3-Channel Rule: Replicating mono spectrogram to RGB
    IN_CHANNELS = 3

    # ==========================================
    # Model Architecture
    # ==========================================
    # List of backbones for the heterogeneous ensemble
    ARCHITECTURES = ["resnet18", "efficientnet_b0", "densenet121"]

    # Data sources to train on
    DATA_SOURCES = [
        "standard",  # Uses SPECTROGRAM_DIR
        "filtered",  # Uses FILTERED_SPECTROGRAM_DIR
    ]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    NUM_WORKERS = 2

    # Optimization
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler
    T_MAX = EPOCHS  # For Cosine Annealing
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 20  # Relaxed patience for Mixup convergence

    # EMA (Exponential Moving Average)
    USE_EMA = True
    EMA_DECAY = 0.95  # Low decay rate for small dataset/few steps

    @classmethod
    def setup(cls):
        """
        Create necessary output directories.
        """
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Ensure directories exist upon import
Config.setup()
