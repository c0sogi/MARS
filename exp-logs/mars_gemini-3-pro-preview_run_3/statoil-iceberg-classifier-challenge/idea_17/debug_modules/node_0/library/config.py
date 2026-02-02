import os


class Config:
    """
    Central configuration for the Aggressively Downsampled Wide-SE-ResNet Ensemble.
    Stores all hyperparameters, file paths, and model architecture settings.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SUBSET_SIZE = 100

    # ==========================================
    # File Paths
    # ==========================================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directories (Write Access)
    # Using 'idea_17' as the specific workspace for this experiment
    WORK_DIR = "./working/idea_17"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Parameters
    # ==========================================
    IMG_HEIGHT = 75
    IMG_WIDTH = 75

    # Input Channels: 3 (Band 1 HH, Band 2 HV, Average (HH+HV)/2)
    IN_CHANNELS = 3

    # Incidence Angle Imputation Strategy
    INC_ANGLE_IMPUTE_STRATEGY = "median"

    # ==========================================
    # Model Architecture
    # ==========================================
    # Custom 4-Stage ResNet with Aggressive Downsampling
    # Channel expansion strategy: 64 -> 128 -> 128 -> 128
    CHANNEL_WIDTHS = [64, 128, 128, 128]

    # Squeeze-and-Excitation Settings
    USE_SE = True
    SE_REDUCTION = 16

    # Classification Head
    # Dropout applied only after activation in the final layer
    DROPOUT_RATE = 0.2
    NUM_CLASSES = 1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    NUM_FOLDS = 5
    NUM_EPOCHS = 50
    BATCH_SIZE = 32

    # Optimization
    # Constant learning rate as per strategy (no scheduler)
    LEARNING_RATE = 1e-3

    # Regularization
    # L2 Weight Decay to prevent confident errors
    WEIGHT_DECAY = 1e-4

    # Early Stopping
    PATIENCE = 10

    # Hardware / Loader Settings
    NUM_WORKERS = 4
    PIN_MEMORY = True

    @classmethod
    def setup(cls):
        """
        Creates the necessary directory structure for caching, checkpoints, and submissions.
        Should be called at the beginning of the execution pipeline.
        """
        directories = [
            cls.WORK_DIR,
            cls.CACHE_DIR,
            cls.CHECKPOINT_DIR,
            cls.SUBMISSION_DIR,
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def get_checkpoint_path(cls, fold_idx):
        """
        Generates the file path for saving the best model checkpoint for a specific fold.

        Args:
            fold_idx (int): The current fold index (0-based).

        Returns:
            str: Full path to the checkpoint file.
        """
        return os.path.join(cls.CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")
