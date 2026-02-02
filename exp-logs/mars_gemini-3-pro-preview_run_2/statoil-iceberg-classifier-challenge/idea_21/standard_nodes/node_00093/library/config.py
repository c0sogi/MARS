import os


class Config:
    """
    Configuration for Quadrant-Pooled Wide-Body Network (QP-WBN).
    Defines constants for paths, data processing, model architecture, and training.
    """

    # ==========================================
    # 1. GENERAL SETTINGS
    # ==========================================
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use when DEBUG is True

    # ==========================================
    # 2. PATHS & DIRECTORIES
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Idea-specific directory
    IDEA_ID = "idea_21"
    IDEA_DIR = os.path.join(WORKING_DIR, IDEA_ID)

    # Artifact sub-directories
    CACHE_DIR = os.path.join(IDEA_DIR, "cache")
    MODEL_DIR = os.path.join(IDEA_DIR, "models")
    SUBMISSION_DIR = os.path.join(IDEA_DIR, "submission")

    # Input Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 3. DATA PARAMETERS
    # ==========================================
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    IN_CHANNELS = 3  # Band 1, Band 2, Average

    # Augmentation
    ROTATION_ANGLES = [0, 90, 180, 270]
    USE_HORIZONTAL_FLIP = True
    USE_VERTICAL_FLIP = False
    USE_MIXUP = False

    # ==========================================
    # 4. MODEL ARCHITECTURE (QP-WBN)
    # ==========================================
    # Wide Backbone Filters to prevent underfitting
    FILTER_SIZES = [64, 128, 128, 128]

    # Regularization
    DROPOUT_RATE = 0.5

    # Readout
    # Quadrant Pooling reduces 4x4 spatial grid to 2x2 quadrants
    READOUT_POOL_SIZE = 2

    # ==========================================
    # 5. TRAINING HYPERPARAMETERS
    # ==========================================
    NUM_FOLDS = 5
    BATCH_SIZE = 32

    # Optimization
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 100

    # Debug overrides
    DEBUG_EPOCHS = 2

    # Scheduler & Early Stopping
    PATIENCE = 15
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.IDEA_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup on import to ensure directories exist
Config.setup()
