import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the Notebook Cell Ordering task.
    Implements the 'Stacked Hybrid with Nearest-Neighbor Semantic Anchoring' strategy.
    """

    # --------------------------------------------------------------------------
    # General Configuration
    # --------------------------------------------------------------------------
    RANDOM_STATE = 42
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SAMPLE_SIZE = 2000  # Number of notebooks to use in debug mode
    NUM_WORKERS = 4  # For data loading

    # --------------------------------------------------------------------------
    # Path Configuration
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea to store cached parquet/joblib files
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Metadata Paths (Pre-generated splits)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Feature Engineering Hyperparameters
    # --------------------------------------------------------------------------
    # TF-IDF Vectorization
    VOCAB_SIZE = 60000
    NGRAM_RANGE = (1, 2)  # Unigrams and Bigrams
    TOKEN_PATTERN = r"(?u)\b\w\w+\b"
    SUBLINEAR_TF = True
    STRIP_ACCENTS = None  # "No Accent Stripping" as per Lesson 4/5

    # Latent Semantic Analysis (LSA)
    SVD_COMPONENTS = 128
    SVD_RANDOM_STATE = 42

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    # Stage 1: Sparse Lexical Regressor (Ridge)
    RIDGE_ALPHA = 1.0
    N_FOLDS = 5  # 5-Fold CV for generating OOF predictions

    # Stage 2: Anchor-Aware Gradient Booster (LightGBM)
    # Minimizing Mean Absolute Error (MAE) for rank prediction
    LGBM_PARAMS = {
        "objective": "regression_l1",  # MAE
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_jobs": -1,
        "verbose": -1,  # Silent execution
        "random_state": 42,
        "max_bin": 255,
    }

    # LightGBM Training Settings
    LGBM_NUM_BOOST_ROUND = 5000
    LGBM_EARLY_STOPPING_ROUNDS = 50
    LGBM_VERBOSE_EVAL = 100

    @classmethod
    def setup(cls):
        """
        Initialize the environment:
        1. Create necessary working and submission directories.
        2. Set random seeds for reproducibility across all libraries.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set fixed random seeds
        random.seed(cls.RANDOM_STATE)
        np.random.seed(cls.RANDOM_STATE)
        torch.manual_seed(cls.RANDOM_STATE)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.RANDOM_STATE)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Configure pandas to display full precision if needed during debugging
        # (Optional, but helps with "print full precision" requirement context)
        import pandas as pd

        pd.set_option("display.precision", 8)
