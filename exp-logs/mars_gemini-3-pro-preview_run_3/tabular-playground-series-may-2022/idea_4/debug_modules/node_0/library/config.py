import os
import torch
import random
import numpy as np


class Config:
    # -------------------------------------------------------------------------
    # Directories and Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Metadata paths (stratified splits)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output paths for model and submission
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache paths for processed data
    TRAIN_PROCESSED_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_PROCESSED_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_PROCESSED_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")
    ENCODERS_PATH = os.path.join(WORKING_DIR, "encoders.npy")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    TARGET_COL = "target"
    ID_COL = "id"
    SEED = 42

    # Feature Definitions
    # Continuous features: f_00 to f_26, f_28
    # Plus the engineered feature 'unique_char_count' derived from f_27
    CONT_FEATURES = [f"f_{i:02d}" for i in range(27)] + ["f_28", "unique_char_count"]

    # Categorical features: f_29, f_30
    # Plus the 10 decomposed character positions from f_27 (f_27_0 to f_27_9)
    CAT_FEATURES = ["f_29", "f_30"] + [f"f_27_{i}" for i in range(10)]

    # -------------------------------------------------------------------------
    # Model Hyperparameters (Gated Funnel MLP)
    # -------------------------------------------------------------------------
    # High-capacity embeddings for categorical variables
    EMBEDDING_DIM = 16

    # Funnel architecture: decreasing width to compress features
    HIDDEN_LAYERS = [512, 256, 128, 64]

    DROPOUT = 0.1
    USE_GLU = True  # Use Gated Linear Units

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 2048
    EPOCHS = 30
    LEARNING_RATE = 1e-2  # Max LR for OneCycleLR
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5  # Early stopping patience
    NUM_WORKERS = 4

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Initialize directories and set random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Configuration loaded. Device: {cls.DEVICE}, Seed: {cls.SEED}")
        print(f"Working Directory: {cls.WORKING_DIR}")
