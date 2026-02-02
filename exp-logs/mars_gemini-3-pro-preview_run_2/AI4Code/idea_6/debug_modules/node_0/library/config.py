import os
import random
import numpy as np
import torch


class Config:
    """
    Central configuration for the Stacked Hybrid Linear-Tree Ranking pipeline.
    """

    # --------------------------------------------------------------------------
    # General Configuration
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data
    NUM_WORKERS = 4

    # --------------------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission Path
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (Parquet format preferred over pickle)
    CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features.parquet")
    CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "val_features.parquet")
    CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "test_features.parquet")

    # Model Artifact Paths
    MODEL_RIDGE_PATH = os.path.join(WORKING_DIR, "ridge_model.joblib")
    MODEL_LGBM_PATH = os.path.join(WORKING_DIR, "lgbm_model.txt")
    VECTORIZER_MD_PATH = os.path.join(WORKING_DIR, "tfidf_md.joblib")
    VECTORIZER_CODE_PATH = os.path.join(WORKING_DIR, "tfidf_code.joblib")
    SVD_MD_PATH = os.path.join(WORKING_DIR, "svd_md.joblib")
    SVD_CODE_PATH = os.path.join(WORKING_DIR, "svd_code.joblib")

    # --------------------------------------------------------------------------
    # Feature Engineering Hyperparameters
    # --------------------------------------------------------------------------
    # Level 1: Markdown TF-IDF (The "Signpost" Features)
    # Settings based on "Idea": Vocab=60k, Bigrams, Logarithmic TF, No Accent Stripping
    MD_TFIDF_PARAMS = {
        "min_df": 2,
        "max_features": 60000,
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "strip_accents": None,
        "lowercase": True,
        "analyzer": "word",
        "token_pattern": r"(?u)\b\w\w+\b",
    }

    # Level 2: Markdown LSA
    MD_SVD_COMPONENTS = 128

    # Level 2: Code Context TF-IDF + LSA
    # Settings based on "Idea": Vocab=5000, Unigrams, SVD=32
    CODE_TFIDF_PARAMS = {
        "min_df": 2,
        "max_features": 5000,
        "ngram_range": (1, 1),
        "sublinear_tf": True,
        "strip_accents": None,
        "lowercase": True,
    }
    CODE_SVD_COMPONENTS = 32

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    # Level 1: Ridge Regression
    RIDGE_ALPHA = 1.0
    N_FOLDS = 5

    # Level 2: LightGBM
    # Minimizing MAE as rank is ordinal/continuous in this formulation
    LGBM_PARAMS = {
        "objective": "regression_l1",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 2000,
        "early_stopping_rounds": 50,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": SEED,
    }

    # --------------------------------------------------------------------------
    # Debugging / Runtime Control
    # --------------------------------------------------------------------------
    DEBUG_SAMPLE_SIZE = 2000  # Number of notebooks to use if DEBUG is True


def setup_reproducibility(seed=Config.SEED):
    """
    Sets random seeds for python, numpy, and torch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in torch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
