import os


class Config:
    """
    Configuration class for the Stacked Hybrid Ranking with Multi-Resolution
    Neighborhood Anchoring pipeline.
    """

    # --------------------------------------------------------------------------
    # Global Settings
    # --------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 12  # Based on available vCPUs

    # Debugging / Development
    # Set to True to run on a small subset of data for rapid iteration
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Cache directory for intermediate files (parquet/npy)
    WORKING_DIR = "./working/idea_20"
    # Output directory for final submission
    SUBMISSION_DIR = "./submission"

    # Specific Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # --------------------------------------------------------------------------
    # Preprocessing Hyperparameters
    # --------------------------------------------------------------------------

    # TF-IDF Vectorizer Settings (Lexical View)
    # High-precision sparse signals
    TFIDF_PARAMS = {
        "input": "content",
        "encoding": "utf-8",
        "decode_error": "replace",
        "strip_accents": None,  # Explicitly None as per "No Accent Stripping"
        "lowercase": True,
        "analyzer": "word",
        "stop_words": "english",
        "token_pattern": r"(?u)\b\w\w+\b",
        "ngram_range": (1, 2),  # Unigrams + Bigrams
        "max_features": 60000,  # Vocabulary size
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "sublinear_tf": True,  # Logarithmic term frequency
    }

    # Truncated SVD Settings (Latent View)
    # Scalable dense semantic space
    SVD_PARAMS = {
        "n_components": 128,
        "algorithm": "randomized",
        "n_iter": 5,
        "random_state": SEED,
    }

    # --------------------------------------------------------------------------
    # Feature Engineering Hyperparameters
    # --------------------------------------------------------------------------

    # Multi-Resolution Anchoring
    # Instance-Based: Specific neighbors to extract explicitly
    ANCHOR_INSTANCE_NEIGHBORS = [1, 2, 3]  # 1st, 2nd, 3rd nearest neighbors

    # Smoothed Neighborhood: Top-K for aggregate stats (Mean/Std)
    ANCHOR_SMOOTHING_K = 5

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------

    # Stage 1: Sparse Lexical Regressor (Ridge)
    # High-bias, low-variance baseline
    N_FOLDS = 5  # For generating OOF predictions
    RIDGE_PARAMS = {"alpha": 1.0, "solver": "auto", "random_state": SEED}

    # Stage 2: Multi-Resolution Gradient Booster (LightGBM)
    # Refinement model minimizing MAE
    LGBM_PARAMS = {
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "regression_l1",  # Minimize MAE
        "metric": "mae",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": SEED,
        "n_jobs": -1,
    }

    # Training Loop Settings
    LGBM_EARLY_STOPPING_ROUNDS = 100
    LGBM_VERBOSE_EVAL = 100

    @classmethod
    def setup(cls):
        """
        Ensure necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
