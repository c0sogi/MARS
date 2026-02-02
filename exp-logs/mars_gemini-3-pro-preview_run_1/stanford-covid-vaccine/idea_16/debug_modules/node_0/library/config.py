import os
import torch


class Config:
    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_16"
    SUBMISSION_DIR = "./submission"

    # Input Metadata Files (Parquet)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Caching Paths (for processed tensors)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    PROCESSED_TRAIN_PATH = os.path.join(CACHE_DIR, "train_data.pt")
    PROCESSED_VAL_PATH = os.path.join(CACHE_DIR, "val_data.pt")
    PROCESSED_TEST_PATH = os.path.join(CACHE_DIR, "test_data.pt")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Targets to be trained on (Scored columns only)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Vocabularies
    # Sequence: A, G, C, U -> Indices 0, 1, 2, 3
    VOCAB_SIZE_SEQ = 4
    # Loop Types: S, M, I, B, H, E, X -> Indices 0..6
    VOCAB_SIZE_LOOP = 7

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Embedding dimensions for atomic sequence and loop type
    EMBEDDING_DIM = 128

    # Backbone: Deep Residual BiGRU
    HIDDEN_DIM = 384
    NUM_LAYERS = 5
    DROPOUT = 0.1
    BIDIRECTIONAL = True

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32  # Adjusted for 384 dim + 107 seq len
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20
    PATIENCE = 5  # Early stopping patience
    MAX_GRAD_NORM = 1.0

    # Debugging / Development
    DEBUG = False  # Set to True to use a small subset of data
    SUBSET_SIZE = 100  # Only used if DEBUG is True

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducible seeds
        import numpy as np
        import random

        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
