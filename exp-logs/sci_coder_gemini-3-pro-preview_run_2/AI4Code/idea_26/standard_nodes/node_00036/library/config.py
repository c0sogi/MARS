import os


class Config:
    """
    Configuration for the Stacked Hybrid Ranking with Decoupled Dual-View Neighborhood Aggregation.
    """

    # --------------------------------------------------------------------------
    # Global Seeding
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_26"
    SUBMISSION_DIR = "./submission"

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Metadata
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache / Artifacts
    # We use these paths to store intermediate results to avoid re-computation
    CACHE_TFIDF_MODEL = os.path.join(WORKING_DIR, "tfidf_vectorizer.joblib")
    CACHE_SVD_MODEL = os.path.join(WORKING_DIR, "svd_model.joblib")

    # Stage 1 Artifacts
    CACHE_RIDGE_MODEL = os.path.join(WORKING_DIR, "stage1_ridge_model.joblib")
    CACHE_STAGE1_OOF = os.path.join(WORKING_DIR, "stage1_oof_preds.parquet")
    CACHE_STAGE1_TEST_PREDS = os.path.join(WORKING_DIR, "stage1_test_preds.parquet")

    # Stage 2 Features
    CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_stage2_features.parquet")
    CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "val_stage2_features.parquet")
    CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "test_stage2_features.parquet")

    # Final Model
    CACHE_LGBM_MODEL = os.path.join(WORKING_DIR, "stage2_lgbm_model.txt")

    # --------------------------------------------------------------------------
    # Feature Engineering Hyperparameters
    # --------------------------------------------------------------------------
    # TF-IDF Vectorizer (The "Signpost" View)
    TFIDF_PARAMS = {
        "input": "content",
        "encoding": "utf-8",
        "decode_error": "replace",
        "strip_accents": None,  # Explicitly keeping accents as per Lesson 00005
        "lowercase": True,
        "analyzer": "word",
        "stop_words": "english",
        "token_pattern": r"(?u)\b\w\w+\b",
        "ngram_range": (1, 2),
        "max_features": 60000,
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "sublinear_tf": True,
    }

    # Truncated SVD (The "Latent" View)
    SVD_N_COMPONENTS = 128
    SVD_RANDOM_STATE = 42

    # Neighborhood Aggregation
    NEIGHBOR_TOP_K = 10  # Number of neighbors to aggregate stats from

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    # Stage 1: Ridge Regression
    RIDGE_PARAMS = {"alpha": 1.0, "solver": "auto", "random_state": 42}

    # Stage 2: LightGBM Regressor
    # Objective is MAE because we are predicting rank positions [0, 1]
    LGBM_PARAMS = {
        "objective": "mae",
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": 42,
        "first_metric_only": True,
    }

    # --------------------------------------------------------------------------
    # Training Loop Settings
    # --------------------------------------------------------------------------
    N_FOLDS = 5
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100

    @classmethod
    def setup(cls):
        """
        Ensures that necessary working directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
