import os
import torch


class Config:
    """
    Central configuration for the RNA Degradation Prediction project.
    Implements the 'Neighborhood-Attentive Structural BiGRU' strategy (Idea 15).
    """

    # ==========================================
    # Project & Experiment Identification
    # ==========================================
    PROJECT_NAME = "RNA_Degradation_Prediction"
    IDEA_NAME = "idea_15"
    SEED = 42

    # ==========================================
    # File Paths & Directories
    # ==========================================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working Directory (Write Allowed)
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Raw Input Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Parquet)
    TRAIN_PARQUET = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PARQUET = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PARQUET = os.path.join(METADATA_DIR, "test.parquet")

    # Outputs
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Cache Paths (for preprocessed tensors)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_cache.npy")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_cache.npy")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_cache.npy")

    # ==========================================
    # Data Dimensions & Features
    # ==========================================
    SEQ_LEN = 107
    SEQ_SCORED = 68
    NUM_TARGETS = 5  # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    # Input Feature Channels (One-Hot Encoded)
    # 4 (Sequence: A,G,U,C) + 3 (Structure: (,.,)) + 7 (Loop: S,M,I,B,H,E,X)
    INPUT_DIM = 14

    # ==========================================
    # Model Architecture (BiGRU + Structural Attention)
    # ==========================================
    # Convolutional Stem
    CONV_FILTERS = 256
    CONV_KERNEL_SIZE = 3

    # Recurrent Backbone
    HIDDEN_DIM = 384
    NUM_LAYERS = 3
    DROPOUT = 0.1

    # Structural Neighborhood Attention
    ATTENTION_WINDOW = 3  # Window size for attending to paired neighbors (j-1, j, j+1)

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Gradient Clipping (Crucial for RNN stability)
    GRAD_CLIP = 1.0

    EPOCHS = 50
    PATIENCE = 10  # Early stopping patience
    NUM_WORKERS = 4

    # Debugging / Development
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100  # Number of samples to use if DEBUG is True

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Initialize the environment:
        1. Create the working directory if it doesn't exist.
        2. Set reproducible seeds.
        """
        # Ensure working directory exists
        os.makedirs(cls.WORKING_DIR, exist_ok=True)

        # Set seeds for reproducibility
        import random
        import numpy as np

        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Automatically setup environment on import
Config.setup()
