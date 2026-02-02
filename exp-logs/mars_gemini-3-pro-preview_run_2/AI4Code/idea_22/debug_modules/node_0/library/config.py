import os


class Config:
    """
    Configuration for the Stacked Hybrid Ranking with Content-Aware Neighbor Projection pipeline.
    """

    # --------------------------------------------------------------------------
    # 1. Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea iteration to prevent conflicts
    WORKING_DIR = "./working/idea_22"

    # Metadata Paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths for Models and Processed Data
    # Using .joblib for sklearn models and .txt for LightGBM
    TFIDF_MODEL_PATH = os.path.join(WORKING_DIR, "tfidf_vectorizer.joblib")
    SVD_MODEL_PATH = os.path.join(WORKING_DIR, "svd_model.joblib")
    RIDGE_MODEL_PATH = os.path.join(WORKING_DIR, "stage1_ridge_model.joblib")
    LGBM_MODEL_PATH = os.path.join(WORKING_DIR, "stage2_lgbm_model.txt")

    # Cache Paths for Dataframes/Arrays (using parquet/npy as requested)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")
    STAGE1_OOF_PATH = os.path.join(WORKING_DIR, "stage1_oof_preds.parquet")
    STAGE1_TEST_PREDS_PATH = os.path.join(WORKING_DIR, "stage1_test_preds.parquet")

    # --------------------------------------------------------------------------
    # 2. Global Settings
    # --------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 4  # For parallel data loading/processing

    # --------------------------------------------------------------------------
    # 3. Vectorization (TF-IDF) Settings
    # --------------------------------------------------------------------------
    # "Vocabulary=60,000, N-gram range=(1, 2), Sublinear TF=True, No Accent Stripping"
    VOCAB_SIZE = 60000
    NGRAM_RANGE = (1, 2)

    TFIDF_PARAMS = {
        "input": "content",
        "encoding": "utf-8",
        "decode_error": "strict",
        "strip_accents": None,  # Explicitly None as per instructions
        "lowercase": True,
        "analyzer": "word",
        "stop_words": "english",
        "token_pattern": r"(?u)\b\w\w+\b",
        "ngram_range": NGRAM_RANGE,
        "max_features": VOCAB_SIZE,
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "sublinear_tf": True,
    }

    # --------------------------------------------------------------------------
    # 4. Latent Space (SVD) Settings
    # --------------------------------------------------------------------------
    # "Truncated SVD (e.g., 128 components)"
    SVD_COMPONENTS = 128

    SVD_PARAMS = {
        "n_components": SVD_COMPONENTS,
        "algorithm": "randomized",
        "n_iter": 5,
        "random_state": SEED,
        "tol": 0.0,
    }

    # --------------------------------------------------------------------------
    # 5. Feature Engineering Settings
    # --------------------------------------------------------------------------
    # "Mean Rank and Std Dev of the Top-K (e.g., K=5)"
    NEIGHBOR_K = 5

    # "LSA (SVD) Vector Components ... (e.g., top 16 dimensions)"
    CONTENT_PROJECTION_DIMS = 16

    # --------------------------------------------------------------------------
    # 6. Model Hyperparameters
    # --------------------------------------------------------------------------

    # Stage 1: Ridge Regression (High-bias linear baseline)
    RIDGE_PARAMS = {
        "alpha": 1.0,
        "fit_intercept": True,
        "copy_X": True,
        "max_iter": None,
        "tol": 0.001,
        "solver": "auto",
        "random_state": SEED,
    }

    # Stage 2: LightGBM Regressor (Content-Aware Refiner)
    # "Minimizing Mean Absolute Error (MAE)"
    LGBM_PARAMS = {
        "objective": "mae",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 5000,  # High number, controlled by early stopping
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 5,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }

    # Training Loop Settings
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE_EVAL = 100

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for the pipeline.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories initialized: {cls.WORKING_DIR}, {cls.SUBMISSION_DIR}")
