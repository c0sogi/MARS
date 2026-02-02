import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Hybrid CNN-BiGRU RNA degradation prediction model.
    Centralizes all hyperparameters, file paths, and configuration settings.
    """

    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True for quick debugging with a subset of data
    DEBUG_SUBSET_SIZE = 100

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"

    # Input Data Paths (using generated metadata)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Specifications
    # -------------------------------------------------------------------------
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Target columns to predict (and compute loss on)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Vocabularies for Integer Encoding
    # Sequence: A, G, U, C
    VOCAB_SEQ = {"A": 0, "G": 1, "U": 2, "C": 3}

    # Structure: (, ), .
    VOCAB_STRUCT = {"(": 0, ")": 1, ".": 2}

    # Predicted Loop Type: S, M, I, B, H, E, X
    VOCAB_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    # Vocabulary Sizes
    SIZE_VOCAB_SEQ = len(VOCAB_SEQ)
    SIZE_VOCAB_STRUCT = len(VOCAB_STRUCT)
    SIZE_VOCAB_LOOP = len(VOCAB_LOOP)

    # -------------------------------------------------------------------------
    # Model Hyperparameters (Hybrid CNN-BiGRU)
    # -------------------------------------------------------------------------
    # Embedding
    EMBED_DIM = 64  # Dimension for each embedding layer (Seq, Struct, Loop)

    # CNN (1D-ResNet) Stage
    CNN_FILTERS = 128
    CNN_KERNEL_SIZE = 3
    CNN_BLOCKS = 3  # Number of residual blocks

    # RNN (Bi-GRU) Stage
    RNN_HIDDEN_DIM = 256
    RNN_LAYERS = 2
    RNN_BIDIRECTIONAL = True

    # Regularization
    DROPOUT = 0.3

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 30
    PATIENCE = 7  # Early stopping patience

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup_directories(cls):
        """Ensures that the working and cache directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)

    @classmethod
    def set_deterministic(cls, seed=None):
        """Sets the random seed for reproducibility."""
        if seed is None:
            seed = cls.SEED

        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Automatically setup directories when imported
Config.setup_directories()
