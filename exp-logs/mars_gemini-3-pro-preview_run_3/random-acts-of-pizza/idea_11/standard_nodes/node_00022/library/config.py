import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the Tri-View Topology-Matched Stacking Ensemble.
    """

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Paths (using pre-split metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Column Definitions
    # ==========================================
    TARGET_COL = "requester_received_pizza"
    ID_COL = "request_id"

    # Text Features
    TEXT_COL = "request_text_edit_aware"
    TITLE_COL = "request_title"

    # Behavioral Features
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # Leakage Prevention
    # Columns ending with this suffix contain future information and must be dropped
    RETRIEVAL_SUFFIX = "_at_retrieval"

    # ==========================================
    # Feature Engineering Parameters
    # ==========================================
    # Lexical View (Sparse)
    TEXT_TFIDF_MAX_FEATURES = 3000
    TEXT_TFIDF_NGRAM_RANGE = (1, 2)

    # Behavioral View (Sparse & Dense)
    SUBREDDIT_TFIDF_MAX_FEATURES = 1000
    SUBREDDIT_SVD_COMPONENTS = 20

    # Semantic View (Dense)
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    RANDOM_SEED = 42
    N_FOLDS = 5

    # Level 1: Lexical & Behavioral Baggers (Random Forest)
    # Optimized for high-dimensional sparse data
    RF_PARAMS = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbose": 0,
    }

    # Level 1: Contextual Booster (XGBoost)
    # Optimized for dense embeddings and latent features
    XGB_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "max_depth": 5,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "enable_categorical": False,
        "early_stopping_rounds": 100,
        "verbosity": 0,
    }

    # Level 2: Meta-Learner (Logistic Regression)
    # Calibrates probabilities from base learners
    META_PARAMS = {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "random_state": RANDOM_SEED,
        "max_iter": 1000,
    }

    # ==========================================
    # Runtime / Debug Options
    # ==========================================
    # Set to True to run on a small subset of data for testing pipeline
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

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
