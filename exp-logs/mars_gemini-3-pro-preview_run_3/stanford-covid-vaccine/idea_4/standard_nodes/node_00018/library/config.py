import os
import torch


class Config:
    """
    Configuration class for the Conformer-based RNA degradation prediction model.
    Encapsulates file paths, data dimensions, model hyperparameters, and training settings.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths (Parquet format)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")

    # Submission Files
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    FINAL_SUBMISSION = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (for deterministic data processing)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data.npy")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data.npy")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data.npy")

    # =========================================================================
    # Data Dimensions and Features
    # =========================================================================
    SEQ_LENGTH = 107
    SCORED_LENGTH = 68  # Number of positions with ground truth

    # Input Channels:
    # 4 (Sequence: A, G, C, U)
    # + 3 (Structure: (, ), .)
    # + 7 (Loop Type: S, M, I, B, H, E, X)
    INPUT_CHANNELS = 14

    # Output Targets:
    # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    OUTPUT_CHANNELS = 5

    # =========================================================================
    # Model Hyperparameters (Conformer Encoder)
    # =========================================================================
    DIM_MODEL = 128
    NUM_HEADS = 4
    NUM_LAYERS = 4
    CONV_KERNEL_SIZE = 7  # Odd number preferred for symmetric padding
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    PATIENCE = 10  # Early stopping patience

    # Device Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # For DataLoader

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    @staticmethod
    def set_seed(seed=42):
        """
        Sets the random seed for reproducibility across all libraries.
        """
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
