import os
import torch


class Config:
    """
    Central configuration for the Iceberg vs Ship classification pipeline.
    Implements the settings for Idea 16: Semi-Supervised SWA-ResNet Ensemble.
    """

    # -------------------------------------------------------------------------
    # General & Reproducibility
    # -------------------------------------------------------------------------
    PROJECT_NAME = "Iceberg_Classifier_SWA_SSL"
    SEED = 42
    NUM_WORKERS = 2  # Optimized for the available 12 vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Input Data (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directory (Artifacts & Cache)
    # Targeting idea_16 as the current iteration
    WORKING_DIR = "./working/idea_16"

    # Sub-directories
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    LOG_DIR = os.path.join(WORKING_DIR, "logs")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing & Normalization
    # -------------------------------------------------------------------------
    # Image Dimensions
    # Upsampling 75x75 -> 224x224 (Bicubic)
    IMG_SIZE = 224

    # Input Channels: 3 (Band 1 Norm, Band 2 Norm, Average of Norms)
    IN_CHANNELS = 3

    # Global Normalization Statistics (Derived from Data Analysis)
    # Used for Min-Max scaling
    BAND1_MIN = -45.5944
    BAND1_MAX = 32.1806

    BAND2_MIN = -45.6555
    BAND2_MAX = 17.8628

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    MODEL_ARCH = "resnet18"
    NUM_CLASSES = 1
    DROPOUT_RATE = 0.5  # For the minimalist head

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    LR = 1e-4
    WEIGHT_DECAY = 0.01

    # Optimization
    MAX_EPOCHS = 100  # Safety cap, actual training relies on convergence detection
    PATIENCE = 10  # For ReduceLROnPlateau or Early Stopping
    SCHEDULER_FACTOR = 0.1

    # Stochastic Weight Averaging (SWA)
    SWA_LR = 1e-5
    SWA_EPOCHS = 12  # Cite solution_lesson_node_00055

    # Ensemble Size
    NUM_MODELS = 5

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    @classmethod
    def setup(cls):
        """
        Initializes the working environment by creating necessary directories.
        Should be called at the start of the pipeline.
        """
        dirs = [
            cls.WORKING_DIR,
            cls.CHECKPOINT_DIR,
            cls.CACHE_DIR,
            cls.LOG_DIR,
            cls.SUBMISSION_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        print(f"Configuration initialized. Working directory: {cls.WORKING_DIR}")
