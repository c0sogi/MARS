import os
import torch


class Config:
    """
    Configuration for the Cardinality-Scaled Wide-Stream Residual BiGRU model.
    Centralizes all file paths, model hyperparameters, and training settings.
    """

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea/experiment
    WORKING_DIR = "./working/idea_38"

    # Input Metadata Files (Parquet)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")

    # Sample Submission
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    # Cache directory for processed tensors
    CACHE_DIR = WORKING_DIR
    # Model checkpoint path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    # Final submission file
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    # Sequence dimensions
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Vocabularies
    VOCAB_SIZE = 4  # A, G, C, U
    LOOP_TYPES = 7  # B, E, H, I, M, S, X

    # Targets to train on (filtering out deg_pH10 and deg_50C)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Caching behavior
    LOAD_CACHED_DATA = True

    # -------------------------------------------------------------------------
    # Model Hyperparameters (Dense Wide-Stream)
    # -------------------------------------------------------------------------
    EMBED_DIM = 100

    # Wide-Stream Backbone
    HIDDEN_DIM = (
        384  # Reduced from 512 to prevent overfitting (Cite solution_lesson_node_00081)
    )

    # Depth
    NUM_LAYERS = 6

    # Regularization
    DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 20

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Stability
    MAX_GRAD_NORM = 1.0

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducibility
        import numpy as np
        import random

        torch.manual_seed(cls.SEED)
        torch.cuda.manual_seed_all(cls.SEED)
        np.random.seed(cls.SEED)
        random.seed(cls.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Initialize directories immediately upon import for safety
Config.setup()
