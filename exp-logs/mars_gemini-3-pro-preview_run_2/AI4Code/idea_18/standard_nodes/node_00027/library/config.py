import os
import random
import numpy as np
import torch
import warnings


class Config:
    """
    Global configuration for the Notebook Cell Ordering task.
    Implements settings for the Stacked Hybrid Ranking with Multi-View Instance-Based Neighborhoods.
    """

    # --------------------------------------------------------------------------
    # 1. Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_18"
    SUBMISSION_DIR = "./submission"

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # 2. General Settings
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to process a smaller subset of data for debugging
    NUM_WORKERS = 4  # Number of workers for parallel data processing

    # --------------------------------------------------------------------------
    # 3. Preprocessing / Feature Engineering
    # --------------------------------------------------------------------------
    # TF-IDF Vectorization Settings
    VOCAB_SIZE = 60000
    NGRAM_RANGE = (1, 2)
    STRIP_ACCENTS = None  # Explicitly None per "No Accent Stripping" lesson
    MIN_DF = 2
    USE_IDF = True
    SUBLINEAR_TF = True

    # Latent Semantic Analysis (SVD)
    SVD_COMPONENTS = 128

    # Symbolic Extraction
    # Regex to extract variables/functions: starts with letter/underscore, followed by alphanum/underscore
    SYMBOLIC_TOKEN_PATTERN = r"[a-zA-Z_][a-zA-Z0-9_]*"

    # Multi-View Instance Neighborhoods
    # Number of neighbors to retrieve for feature extraction (Lexical, Latent, Symbolic views)
    NUM_NEIGHBORS = 5

    # --------------------------------------------------------------------------
    # 4. Model Hyperparameters
    # --------------------------------------------------------------------------
    # Stage 1: Sparse Lexical Regressor (Ridge)
    RIDGE_ALPHA = 1.0

    # Stage 2: Multi-View Instance Gradient Booster (LightGBM)
    # Objective is MAE (Mean Absolute Error) to align with rank prediction
    LGBM_PARAMS = {
        "objective": "regression_l1",  # MAE
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,  # Silent execution
        "random_state": SEED,
        "n_jobs": -1,
    }

    # Training Loop Settings
    LGBM_EARLY_STOPPING_ROUNDS = 50
    LGBM_VERBOSE_EVAL = 100

    @classmethod
    def setup(cls):
        """
        Creates necessary working and submission directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def set_seeds(cls):
        """
        Sets random seeds for python, numpy, and torch to ensure reproducibility.
        """
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
        # Ensure deterministic behavior where possible
        os.environ["PYTHONHASHSEED"] = str(cls.SEED)

        # Suppress warnings for cleaner output
        warnings.filterwarnings("ignore")
