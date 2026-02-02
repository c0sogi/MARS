import os
import torch


class Config:
    """
    Global configuration for the Apple Disease Detection task.
    Implements the 'Proxy-Validated Full-Data Training' strategy settings.
    """

    # ==========================================
    # System Settings
    # ==========================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 12 vCPUs available; 4 workers is a robust default for data loading
    NUM_WORKERS = 4

    # ==========================================
    # Directory & File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Image Directory
    IMAGE_DIR = os.path.join(INPUT_DIR, "images")

    # Raw Data (Read-Only)
    RAW_TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    RAW_TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Generated Pre-processing)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Directories (Writeable)
    # Using 'idea_10' as the specific workspace for this strategy
    WORKING_DIR = "./working/idea_10"
    OUTPUT_DIR = os.path.join(WORKING_DIR, "output")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 256
    BATCH_SIZE = 32
    NUM_CLASSES = 4

    # Target columns must match the submission format order
    TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

    # ==========================================
    # Model Configuration
    # ==========================================
    MODEL_NAME = "resnet34"
    PRETRAINED = True

    # ==========================================
    # Training Configuration
    # ==========================================
    # Phase 1: Calibration (Finding optimal epochs via CV)
    EPOCHS_CALIBRATION = 20
    N_FOLDS = 5

    # Optimizer Settings
    LR = 1e-4
    WEIGHT_DECAY = 1e-5

    # Scheduler Settings
    MIN_LR = 1e-6

    @classmethod
    def setup(cls):
        """
        Initialize the workspace by creating necessary directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
