import os


class Config:
    """
    Configuration for Stacked Hybrid Ranking with Uncertainty-Aware Multi-View Anchoring.
    """

    # --------------------------------------------------------------------------
    # General Configuration
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to process a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 2000
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist immediately
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Metadata
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cached Artifacts (Transformers & Models)
    CACHE_TFIDF_VECTORIZER = os.path.join(WORKING_DIR, "tfidf_vectorizer.joblib")
    CACHE_SVD_MODEL = os.path.join(WORKING_DIR, "svd_model.joblib")
    CACHE_STAGE1_RIDGE = os.path.join(WORKING_DIR, "stage1_ridge.joblib")
    CACHE_STAGE2_LGBM = os.path.join(WORKING_DIR, "stage2_lgbm.txt")

    # Cached Processed Data (Parquet)
    CACHE_TRAIN_PROCESSED = os.path.join(WORKING_DIR, "train_processed.parquet")
    CACHE_VAL_PROCESSED = os.path.join(WORKING_DIR, "val_processed.parquet")
    CACHE_TEST_PROCESSED = os.path.join(WORKING_DIR, "test_processed.parquet")

    # --------------------------------------------------------------------------
    # Feature Engineering Hyperparameters
    # --------------------------------------------------------------------------
    # TF-IDF Vectorization
    TFIDF_PARAMS = {
        "input": "content",
        "encoding": "utf-8",
        "decode_error": "replace",
        "strip_accents": None,  # Preserving accents as per lessons
        "lowercase": True,
        "analyzer": "word",
        "stop_words": "english",
        "token_pattern": r"(?u)\b\w\w+\b",
        "ngram_range": (1, 2),
        "max_features": 60000,
        "sublinear_tf": True,
        "use_idf": True,
    }

    # Latent Semantic Analysis (SVD)
    SVD_N_COMPONENTS = 128
    SVD_RANDOM_STATE = SEED

    # Neighborhood Aggregation
    # Number of neighbors (Top-K) to consider for calculating rank statistics (mean, std)
    NEIGHBOR_K = 20

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    # Stage 1: Ridge Regression (The "Signpost" Model)
    RIDGE_ALPHA = 1.0
    RIDGE_SOLVER = "auto"

    # Stage 2: LightGBM Regressor (The "Refinement" Model)
    LGBM_PARAMS = {
        "n_estimators": 10000,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "objective": "mae",  # Minimize Mean Absolute Error for rank prediction
        "metric": "mae",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": -1,
        "importance_type": "gain",
    }

    # Training Loop Settings
    LGBM_EARLY_STOPPING_ROUNDS = 100
    LGBM_VERBOSE_EVAL = 250
