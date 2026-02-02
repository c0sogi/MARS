import os


class Config:
    # ==========================================
    # General Configuration
    # ==========================================
    SEED = 42
    # Flag to enable debugging mode with a smaller dataset
    DEBUG = False
    # Number of samples to use when DEBUG is True
    DEBUG_SAMPLE_SIZE = 100

    # ==========================================
    # Data Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Raw JSON files (read-only)
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Generated Metadata CSVs
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Working Directory & Artifacts
    # ==========================================
    # Specific working directory for this idea iteration
    WORKING_DIR = "./working/idea_23"

    # Cache file for processed tensors (numpy format)
    CACHE_FILE = "processed_data.npz"
    CACHE_PATH = os.path.join(WORKING_DIR, CACHE_FILE)

    # Directory to save model checkpoints
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Input dimensions
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    # 3 Channels: Band 1, Band 2, Mean(Band 1, Band 2)
    IN_CHANNELS = 3

    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 2e-4  # "Low and Slow" strategy

    # Regularization
    DROPOUT_RATE = 0.5
    WEIGHT_DECAY = 1e-4

    # Training Strategy
    NUM_FOLDS = 5
    PATIENCE = 10  # Early stopping patience

    # ==========================================
    # Utility Methods
    # ==========================================
    @classmethod
    def create_directories(cls):
        """Creates necessary working and submission directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories created/verified at: {cls.WORKING_DIR}")

    @classmethod
    def get_checkpoint_path(cls, fold_idx):
        """Returns the path for a specific fold's model checkpoint."""
        return os.path.join(cls.CHECKPOINT_DIR, f"wbmg_net_fold_{fold_idx}.pth")
