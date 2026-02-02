import os
import torch


class Config:
    """
    Configuration for the Aggregate-Residual Parallel Funnel Ensemble (ARPFE) strategy.
    Defines hyperparameters, file paths, and model architecture specifications.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Data Paths
    # ==========================================
    # Input data (Metadata generated splits)
    TRAIN_DATA_PATH = "./metadata/train.csv"
    VAL_DATA_PATH = "./metadata/val.csv"
    TEST_DATA_PATH = "./metadata/test.csv"

    # Output directories
    # We use a specific directory for this idea to store cached processed data
    CACHE_DIR = "./working/idea_34/"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Submission file
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 1024
    EPOCHS = 50
    MAX_LR = 1e-2
    WEIGHT_DECAY = 1e-5

    # ==========================================
    # Feature Engineering Configuration
    # ==========================================
    # Continuous features: f_00 to f_26, plus f_28
    CONT_FEATURES = [f"f_{i:02d}" for i in range(29) if i != 27]

    # Categorical features in raw data (excluding f_27 which is special)
    # f_29 and f_30 are treated as categorical
    BASE_CAT_FEATURES = ["f_29", "f_30"]

    # f_27 Decomposition
    # We decompose the string into 10 fixed positions
    F27_LENGTH = 10
    F27_PREFIX = "f_27_char"

    # Generated Aggregate Features (Set-theoretic properties)
    AGG_FEATURES = [
        "unique_character_count",
        "max_char_frequency",
        "min_char_frequency",
    ]

    # Embedding Dimension for all categorical features
    EMBEDDING_DIM = 16

    # ==========================================
    # Model Architecture (ARPFE)
    # ==========================================
    # 5 Independent Streams with Heterogeneous Configuration
    # Format: {'layers': [List of widths], 'dropout': float}

    STREAM_CONFIGS = [
        # Stream 1 (Anchor): Standard Funnel
        {"layers": [512, 256, 128], "dropout": 0.20},
        # Stream 2 (Anchor): Standard Funnel
        {"layers": [512, 256, 128], "dropout": 0.20},
        # Stream 3 (Capacity Variant): Wide Funnel
        {"layers": [1024, 512, 256], "dropout": 0.25},
        # Stream 4 (Capacity Variant): Wide Funnel
        {"layers": [1024, 512, 256], "dropout": 0.25},
        # Stream 5 (Conservative): Standard Funnel, Higher Regularization
        {"layers": [512, 256, 128], "dropout": 0.30},
    ]

    @classmethod
    def get_all_cat_features(cls):
        """Returns list of all categorical feature names after processing."""
        f27_cols = [f"{cls.F27_PREFIX}_{i}" for i in range(cls.F27_LENGTH)]
        return cls.BASE_CAT_FEATURES + f27_cols

    @classmethod
    def get_all_cont_features(cls):
        """Returns list of all continuous feature names after processing."""
        return cls.CONT_FEATURES + cls.AGG_FEATURES
