import os


class Config:
    # --------------------------------------------------------------------------
    # 1. Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary write directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # --------------------------------------------------------------------------
    # 2. Global Configuration
    # --------------------------------------------------------------------------
    RANDOM_STATE = 42
    NUM_WORKERS = 4

    # Debugging: Set to True to use a smaller subset of data for testing
    DEBUG = False
    DEBUG_SAMPLES = 1000  # Number of samples if DEBUG is True

    # --------------------------------------------------------------------------
    # 3. Feature Engineering Parameters
    # --------------------------------------------------------------------------
    # TF-IDF Vectorization (Lexical View)
    TFIDF_PARAMS = {
        "input": "content",
        "encoding": "utf-8",
        "decode_error": "strict",
        "strip_accents": None,  # Explicitly None as per instructions
        "lowercase": True,
        "preprocessor": None,
        "tokenizer": None,
        "analyzer": "word",
        "stop_words": None,
        "token_pattern": r"(?u)\b\w\w+\b",
        "ngram_range": (1, 2),
        "max_df": 1.0,
        "min_df": 1,
        "max_features": 60000,
        "vocabulary": None,
        "binary": False,
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "sublinear_tf": True,
    }

    # Truncated SVD (Latent View)
    SVD_PARAMS = {
        "n_components": 128,
        "algorithm": "randomized",
        "n_iter": 5,
        "random_state": RANDOM_STATE,
        "tol": 0.0,
    }

    # --------------------------------------------------------------------------
    # 4. Model Hyperparameters
    # --------------------------------------------------------------------------
    # Stage 1: Ridge Regression (Sparse Lexical Signpost)
    RIDGE_PARAMS = {
        "alpha": 1.0,
        "fit_intercept": True,
        "copy_X": True,
        "max_iter": None,
        "tol": 0.001,
        "solver": "auto",
        "random_state": RANDOM_STATE,
    }

    # Stage 2: LightGBM (Dual-View Anchor Ranking)
    LGBM_PARAMS = {
        "objective": "regression_l1",  # Minimize MAE
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
        # Note: early_stopping_rounds is passed to .train() or .fit(),
        # but we define the value here for reference
        "early_stopping_rounds": 100,
    }

    # --------------------------------------------------------------------------
    # 5. Training Configuration
    # --------------------------------------------------------------------------
    N_FOLDS = 5

    # --------------------------------------------------------------------------
    # 6. Cache File Paths
    # --------------------------------------------------------------------------
    # Dataframes
    TRAIN_FEATS_PATH = os.path.join(CACHE_DIR, "train_features.parquet")
    VAL_FEATS_PATH = os.path.join(CACHE_DIR, "val_features.parquet")
    TEST_FEATS_PATH = os.path.join(CACHE_DIR, "test_features.parquet")

    # Artifacts
    TFIDF_MODEL_PATH = os.path.join(CACHE_DIR, "tfidf_vectorizer.joblib")
    SVD_MODEL_PATH = os.path.join(CACHE_DIR, "svd_model.joblib")
    RIDGE_MODEL_PATH = os.path.join(CACHE_DIR, "ridge_model.joblib")
    LGBM_MODEL_PATH = os.path.join(CACHE_DIR, "lgbm_model.txt")

    # Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
