import os
import torch


class Config:
    """
    Global configuration for the Tweet Sentiment Extraction task.
    Implements parameters for DeBERTa-v3-Large with Dual-Head Multi-Task Learning.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    N_FOLDS = 5
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True

    # =========================================================================
    # Data Paths
    # =========================================================================
    METADATA_DIR = "./metadata"
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
    VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
    TEST_FILE = os.path.join(METADATA_DIR, "test.csv")

    # Output and Cache Paths
    WORKING_DIR = "./working/idea_5"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Model Configuration
    # =========================================================================
    MODEL_PATH = "microsoft/deberta-v3-large"
    TOKENIZER_PATH = "microsoft/deberta-v3-large"
    MAX_LEN = 96

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 3
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    DROPOUT = 0.1

    # Loss Configuration
    LABEL_SMOOTHING = 0.1
    CONTENT_LOSS_WEIGHT = 0.5  # Lambda for the auxiliary content segmentation task

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2

    @classmethod
    def setup(cls, debug=False, epochs=None, batch_size=None):
        """
        Initializes the configuration, creating necessary directories and
        applying runtime overrides.

        Args:
            debug (bool): If True, enables debug mode (smaller dataset).
            epochs (int, optional): Override default number of epochs.
            batch_size (int, optional): Override default batch size.
        """
        cls.DEBUG = debug

        if epochs is not None:
            cls.EPOCHS = epochs

        if batch_size is not None:
            cls.TRAIN_BATCH_SIZE = batch_size

        # Create necessary directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        print(f"Configuration Setup Complete:")
        print(f"  Model: {cls.MODEL_PATH}")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Epochs: {cls.EPOCHS}")
        print(f"  Batch Size: {cls.TRAIN_BATCH_SIZE}")
        print(f"  Debug Mode: {cls.DEBUG}")
        print(f"  Cache Directory: {cls.CACHE_DIR}")
