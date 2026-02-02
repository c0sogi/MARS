import os
import torch


class Config:
    """
    Centralized configuration for the DRHA-CNN experiment.
    """

    # Experiment Identifier
    EXP_ID = "idea_36"

    # Reproducibility
    SEED = 42

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    BASE_DIR = os.getcwd()
    INPUT_DIR = os.path.join(BASE_DIR, "input")
    METADATA_DIR = os.path.join(BASE_DIR, "metadata")

    # Working directory for this specific experiment idea
    WORKING_DIR = os.path.join(BASE_DIR, "working", EXP_ID)

    # Cache directory for processed numpy arrays (deterministic data processing)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Checkpoint directory for model weights
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Submission directory
    SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Raw JSON files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata CSVs (Pre-generated)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Output Submission File
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Parameters
    # -------------------------------------------------------------------------
    IMG_WIDTH = 75
    IMG_HEIGHT = 75
    IN_CHANNELS = 3  # Band 1 (HH), Band 2 (HV), Average ((HH+HV)/2)
    NUM_CLASSES = 1  # Binary classification (Iceberg vs Ship)

    # Debugging / Development
    DEBUG = False  # Set to True to limit dataset size for quick testing
    MAX_DEBUG_SAMPLES = 100  # Number of samples to use if DEBUG is True

    # -------------------------------------------------------------------------
    # Model Architecture (DRHA-CNN)
    # -------------------------------------------------------------------------
    # Channel widths for the 4-stage Plain CNN
    CHANNEL_CONFIG = [64, 128, 128, 128]

    # Regularization
    DROPBLOCK_PROB = 0.1  # Target drop probability for DropBlock
    DROPBLOCK_BLOCK_SIZE = 5  # Size of the square block to drop (approx 75/15)
    DROPOUT_RATE = 0.5  # Standard dropout in the classification head

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    N_FOLDS = 5
    EPOCHS = 100
    BATCH_SIZE = 64

    # Optimizer
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    # Early Stopping
    PATIENCE = 15

    # -------------------------------------------------------------------------
    # Compute
    # -------------------------------------------------------------------------
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates the necessary directory structure for the experiment.
        Should be called at the start of the execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        if cls.DEBUG:
            print(
                f"[{cls.EXP_ID}] Running in DEBUG mode with max {cls.MAX_DEBUG_SAMPLES} samples."
            )
