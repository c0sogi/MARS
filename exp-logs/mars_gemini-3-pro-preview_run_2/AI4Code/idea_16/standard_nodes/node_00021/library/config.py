import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the Stacked Hybrid Ranking Pipeline.
    """

    # --------------------------------------------------------------------------
    # 1. Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate files (Parquet/NPY/Models)
    WORKING_DIR = "./working/idea_16"

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # 2. Data Processing & Feature Engineering
    # --------------------------------------------------------------------------
    # Text Vectorization (TF-IDF)
    MD_VOCAB_SIZE = 60000
    NGRAM_RANGE = (1, 2)

    # Latent Semantic Analysis (SVD)
    SVD_COMPONENTS = 128

    # Distributional Anchoring
    # Number of bins to discretize the notebook length [0, 1] for density histograms
    N_BINS = 10

    # Top-K Smoothing
    TOP_K = 20  # Number of neighbors to consider for simple stats

    # --------------------------------------------------------------------------
    # 3. Model Hyperparameters
    # --------------------------------------------------------------------------
    # Stage 1: Ridge Regression
    RIDGE_ALPHA = 1.0

    # Stage 2: LightGBM
    LGBM_PARAMS = {
        "objective": "mae",  # Minimize Mean Absolute Error
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": 1000,  # Maximum rounds
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,  # Silent
        "n_jobs": 4,
        "random_state": 42,
    }

    # --------------------------------------------------------------------------
    # 4. Runtime Settings
    # --------------------------------------------------------------------------
    RANDOM_STATE = 42
    DEBUG = False  # Set to True to run on a small subset for testing
    DEBUG_SAMPLE_SIZE = 1000

    # Validation
    NUM_FOLDS = 5
    EARLY_STOPPING_ROUNDS = 50

    @classmethod
    def setup(cls):
        """
        Initialize the environment:
        1. Create necessary directories.
        2. Set random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.RANDOM_STATE)
        np.random.seed(cls.RANDOM_STATE)
        torch.manual_seed(cls.RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.RANDOM_STATE)

        # Ensure LightGBM determinism via params (handled in model init usually,
        # but setting os environ helps too)
        os.environ["PYTHONHASHSEED"] = str(cls.RANDOM_STATE)

    @classmethod
    def get_lgbm_params(cls):
        """Returns a copy of LGBM params with the configured random state."""
        params = cls.LGBM_PARAMS.copy()
        params["random_state"] = cls.RANDOM_STATE
        return params
