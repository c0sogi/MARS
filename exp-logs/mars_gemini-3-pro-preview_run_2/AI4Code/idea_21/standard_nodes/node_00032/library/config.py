import os


class Config:
    """
    Global configuration for the Notebook Cell Ordering task.
    Implements settings for the Stacked Hybrid Ranking with Multi-Resolution Neighborhood Anchoring.
    """

    # --------------------------------------------------------------------------
    # 1. Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate files (parquet/npy)
    # Specific to this 'idea_22' iteration
    WORKING_DIR = "./working/idea_22"

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # 2. Global Settings
    # --------------------------------------------------------------------------
    RANDOM_STATE = 42
    NUM_WORKERS = 4  # Number of parallel workers for data loading/processing

    # --------------------------------------------------------------------------
    # 3. Feature Engineering Hyperparameters
    # --------------------------------------------------------------------------

    # TF-IDF Vectorizer (The "Lexical" View)
    # Settings: Vocab=60k, N-gram=(1,2), Sublinear TF, No Accent Stripping
    TFIDF_PARAMS = {
        "input": "content",
        "encoding": "utf-8",
        "decode_error": "replace",
        "strip_accents": None,  # Explicitly preserve accents
        "lowercase": True,
        "analyzer": "word",
        "stop_words": "english",  # Standard english stop words
        "token_pattern": r"(?u)\b\w\w+\b",
        "ngram_range": (1, 2),
        "max_features": 60000,
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "sublinear_tf": True,
    }

    # Truncated SVD (The "Latent" View)
    # Settings: 128 Components on top of TF-IDF
    SVD_PARAMS = {
        "n_components": 128,
        "algorithm": "randomized",
        "n_iter": 5,
        "random_state": RANDOM_STATE,
    }

    # Multi-Resolution Anchoring
    # Settings: Top-10 neighbors for smoothing
    ANCHOR_PARAMS = {"top_k": 10, "resolutions": ["lexical", "latent"]}

    # --------------------------------------------------------------------------
    # 4. Model Hyperparameters
    # --------------------------------------------------------------------------

    # Stage 1: Sparse Lexical Regressor (Ridge)
    # 5-Fold CV for OOF generation
    N_FOLDS_STAGE1 = 5

    RIDGE_PARAMS = {"alpha": 1.0, "solver": "auto", "random_state": RANDOM_STATE}

    # Stage 2: Multi-Resolution Gradient Booster (LightGBM)
    # Minimizing MAE
    LGBM_PARAMS = {
        "objective": "mae",
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": RANDOM_STATE,
    }

    # Early stopping for Stage 2
    EARLY_STOPPING_ROUNDS = 100

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories if they do not exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup()
