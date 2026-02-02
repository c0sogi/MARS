import os
import torch


class Config:
    # ==========================================
    # Global Configuration
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use when DEBUG is True

    # ==========================================
    # Compute Configuration
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of CPU workers for data loading

    # ==========================================
    # Data Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    # The bagging strategy requires the full dataset.
    # Downstream scripts should combine train and val metadata.
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Images
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # ==========================================
    # Output Paths
    # ==========================================
    # Working directory for this specific idea/iteration
    WORKING_DIR = "./working/idea_18"

    # Sub-directories for organization
    OUTPUT_DIR = os.path.join(WORKING_DIR, "output")
    MODELS_DIR = os.path.join(WORKING_DIR, "models")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "resnet34"
    NUM_CLASSES = 4
    PRETRAINED = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Seed Averaging Strategy
    NUM_SEEDS = 5  # Number of independent models to train

    # Input dimensions
    IMAGE_SIZE = 256

    # Optimization
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Scheduler: Cosine Annealing Warm Restarts
    # T_0 is synchronized with EPOCHS to ensure full decay
    T_0 = 15
    T_MULT = 1
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 5

    # Loss Weights (Optional, calculated dynamically usually, but placeholders can be here)
    USE_CLASS_WEIGHTS = True

    @classmethod
    def setup(cls):
        """
        Creates the necessary directory structure for the experiment.
        This should be called at the beginning of the pipeline.
        """
        dirs = [
            cls.WORKING_DIR,
            cls.OUTPUT_DIR,
            cls.MODELS_DIR,
            cls.CACHE_DIR,
            cls.SUBMISSION_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

        if cls.DEBUG:
            print(f"Configuration: DEBUG Mode Enabled (Size: {cls.DEBUG_SAMPLE_SIZE})")

    def __init__(self, **kwargs):
        """
        Initialize with optional overrides for hyperparameters.
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
