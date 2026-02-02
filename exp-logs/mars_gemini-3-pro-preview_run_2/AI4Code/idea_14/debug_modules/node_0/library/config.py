import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the Notebook Cell Ordering pipeline.
    Implements the 'Stacked Hybrid Ranking with Multi-View Symbolic and Semantic Anchoring'.
    """

    # --------------------------------------------------------------------------
    # 1. Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_14"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission Path
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # 2. Data Preprocessing & Vectorization
    # --------------------------------------------------------------------------
    VOCAB_SIZE = 60000
    NGRAM_RANGE = (1, 2)
    # Regex for tokenization (sklearn default)
    TOKEN_PATTERN = r"(?u)\b\w\w+\b"

    # Dimensionality Reduction (Latent Semantic Analysis)
    SVD_COMPONENTS = 128
    SVD_RANDOM_STATE = 42

    # --------------------------------------------------------------------------
    # 3. Feature Engineering (Anchors)
    # --------------------------------------------------------------------------
    # Number of neighbors to consider for anchor features
    ANCHOR_K = 20

    # --------------------------------------------------------------------------
    # 4. Model Hyperparameters
    # --------------------------------------------------------------------------
    # Stage 1: Ridge Regression (The "Signpost" Model)
    RIDGE_ALPHA = 1.0
    RIDGE_RANDOM_STATE = 42

    # Stage 2: LightGBM (The "Refinement" Model)
    LGBM_PARAMS = {
        "objective": "mae",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": 42,
        "n_estimators": 2000,
        "early_stopping_rounds": 100,
    }

    # --------------------------------------------------------------------------
    # 5. Utilities
    # --------------------------------------------------------------------------
    RANDOM_SEED = 42

    @staticmethod
    def set_seed(seed=42):
        """
        Sets the random seed for reproducibility across Python, NumPy, and Torch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
