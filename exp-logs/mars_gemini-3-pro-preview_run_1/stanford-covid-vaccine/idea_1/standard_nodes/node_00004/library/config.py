import os
import torch
import numpy as np
import random


class Config:
    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for processed data (Idea 1)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    # Submission output path
    SUBMISSION_FILE = os.path.join(WORKING_DIR, "submission.csv")

    # Metadata File Paths
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.parquet")
    VAL_FILE = os.path.join(METADATA_DIR, "val.parquet")
    TEST_FILE = os.path.join(METADATA_DIR, "test.parquet")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    SEQ_SCORED = 68

    # Columns to predict (Submission requires all 5)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Columns actually used for scoring in the competition metric
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # =========================================================================
    # Tokenization Mappings
    # =========================================================================
    # 0-based indexing for Embeddings

    # Sequence: A, G, C, U
    SEQ_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}

    # Structure: (, ), .
    STRUCT_MAP = {"(": 0, ")": 1, ".": 2}

    # Predicted Loop Type: S, M, I, B, H, E, X
    LOOP_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    # Vocabulary sizes for Embeddings
    VOCAB_SIZE_SEQ = len(SEQ_MAP)
    VOCAB_SIZE_STRUCT = len(STRUCT_MAP)
    VOCAB_SIZE_LOOP = len(LOOP_MAP)

    # =========================================================================
    # Model Hyperparameters (1D-CNN Baseline)
    # =========================================================================
    EMBED_DIM = 32  # Dimension for input embeddings
    HIDDEN_DIM = 128  # Hidden dimension for GRU
    FILTER_CHANNELS = 128  # Number of filters in Conv layers
    KERNEL_SIZE = 3  # Kernel size for Conv layers
    DROPOUT = 0.3  # Dropout rate
    LAYERS = 3  # Number of Conv blocks

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    PATIENCE = 10  # Early stopping patience
    SEED = 42

    # Debugging / Development
    DEBUG = False  # Set to True to use a smaller subset of data
    DEBUG_SAMPLES = 100  # Number of samples if DEBUG is True

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    @staticmethod
    def setup_directories():
        """Creates necessary working directories."""
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

    @staticmethod
    def set_seed(seed=None):
        """Sets the random seed for reproducibility."""
        if seed is None:
            seed = Config.SEED

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
