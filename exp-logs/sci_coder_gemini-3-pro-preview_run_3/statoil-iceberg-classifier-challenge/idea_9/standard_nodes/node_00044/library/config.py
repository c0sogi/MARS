import os
import torch


class Config:
    """
    Configuration for the Spatially-Regularized Narrow ResNet (SRN-ResNet) pipeline.
    Centralizes hyperparameters, file paths, and compute settings.
    """

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing logic
    NUM_WORKERS = 2  # Number of data loading workers

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"

    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Generated previously)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing
    # -------------------------------------------------------------------------
    IMAGE_SIZE = 75
    IN_CHANNELS = 3  # HH, HV, Average(HH, HV)

    # -------------------------------------------------------------------------
    # Model Architecture (SRN-ResNet)
    # -------------------------------------------------------------------------
    # Backbone constraints
    STEM_FILTERS = 64
    STAGE_FILTERS = [64, 64, 128]  # Strictly capped width

    # Regularization
    SPATIAL_DROPOUT_RATE = 0.1  # Structural regularization inside blocks
    CLASSIFIER_DROPOUT_RATE = 0.2  # Dropout before final layer

    # Head
    CLASSIFIER_HIDDEN_DIM = 512  # Single hidden layer width

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    N_FOLDS = 5
    BATCH_SIZE = 32
    EPOCHS = 40

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # L2 Regularization strictly enforced

    # Early Stopping
    PATIENCE = 10

    # -------------------------------------------------------------------------
    # Compute
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_checkpoint_path(cls, fold):
        """
        Returns the path for saving the best model checkpoint for a specific fold.
        """
        return os.path.join(cls.WORKING_DIR, f"model_fold_{fold}.pth")


# Execute setup on import to ensure directories are ready
Config.setup()
