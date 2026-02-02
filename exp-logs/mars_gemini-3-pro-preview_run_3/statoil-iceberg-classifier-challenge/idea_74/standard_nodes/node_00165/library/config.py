import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Directories and Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Main working directory for this specific idea/experiment
    WORK_DIR = "./working/idea_74"

    # Subdirectories for organization
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORK_DIR, "submission")

    # Input Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files
    METADATA_TRAIN = os.path.join(METADATA_DIR, "train.csv")
    METADATA_VAL = os.path.join(METADATA_DIR, "val.csv")
    METADATA_TEST = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Global Settings
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on vCPU count (12 vCPUs available)

    # -------------------------------------------------------------------------
    # Data Hyperparameters
    # -------------------------------------------------------------------------
    IMAGE_SIZE = 75
    IN_CHANNELS = 3  # HH, HV, Average

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    NUM_FOLDS = 5
    NUM_EPOCHS = 75
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    PATIENCE = 12
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    # -------------------------------------------------------------------------
    # Model Architecture Settings
    # -------------------------------------------------------------------------
    # Backbone settings
    Use_Bias = True
    Negative_Slope = 0.1  # For LeakyReLU

    # Attention settings
    SE_Reduction = 16

    # Readout/Calibration settings
    Feature_Dim = 128  # Dimension of V3 and V4 vectors
    Calibration_Dim = 32  # Hidden dimension for angle encoder

    # Classification Head
    Dropout_Rate = 0.5

    # Inference
    USE_TTA = False  # Explicitly disable Test-Time Augmentation

    @classmethod
    def setup_directories(cls):
        """Creates necessary working directories if they don't exist."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
