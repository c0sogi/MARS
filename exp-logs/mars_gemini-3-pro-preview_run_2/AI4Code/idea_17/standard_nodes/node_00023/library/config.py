import os


class Config:
    """
    Global configuration for the Stacked Hybrid Ranking with Functional Landmark Triangulation pipeline.
    """

    # --------------------------------------------------------------------------
    # 1. Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Directory for intermediate files (Parquet/NPY)
    # Using specific directory for Idea 17 to avoid conflicts
    WORKING_DIR = "./working/idea_17"

    # Output Directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # 2. Global Settings
    # --------------------------------------------------------------------------
    RANDOM_STATE = 42
    NUM_JOBS = 12  # Utilizing available vCPUs

    # --------------------------------------------------------------------------
    # 3. Preprocessing Hyperparameters
    # --------------------------------------------------------------------------
    # Stage 1 & 2 Text Vectorization (Markdown)
    # "Vocab=60,000, N-gram range=(1, 2), Sublinear TF=True"
    MD_TFIDF_PARAMS = {
        "min_df": 2,
        "max_df": 0.9,
        "max_features": 60000,
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "strip_accents": None,  # Explicitly keeping accents as per lessons
        "use_idf": True,
        "smooth_idf": True,
        "stop_words": "english",  # Optional, but often helpful
    }

    # Functional Landmark Generation (Code Clustering)
    # "Fit a separate TF-IDF + Truncated SVD pipeline on the aggregated Code Cells"
    CODE_TFIDF_PARAMS = {
        "min_df": 2,
        "max_features": 20000,  # Smaller vocab for code syntax/keywords
        "sublinear_tf": True,
    }
    CODE_SVD_COMPONENTS = 64
    NUM_CODE_CLUSTERS = 5  # "Mini-Batch K-Means (e.g., K=5)"

    # Neighborhood Smoothing
    NEIGHBORHOOD_SIZE = 5  # Top-N code cells for mean rank calculation

    # --------------------------------------------------------------------------
    # 4. Model Hyperparameters
    # --------------------------------------------------------------------------
    # Stage 1: Sparse Lexical Regressor (Ridge)
    RIDGE_ALPHA = 1.0
    NUM_FOLDS = 5  # For generating unbiased OOF predictions for Stage 2

    # Stage 2: Landmark-Aware Gradient Booster (LightGBM)
    # "Minimizing Mean Absolute Error (MAE)"
    LGBM_PARAMS = {
        "objective": "regression_l1",  # MAE
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 5000,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_jobs": -1,
        "verbosity": -1,
        "random_state": RANDOM_STATE,
    }

    # Training Loop Settings
    LGBM_EARLY_STOPPING_ROUNDS = 100
    LGBM_VERBOSE_EVAL = 100

    @classmethod
    def setup(cls):
        """
        Creates necessary working directories if they do not exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories on import
Config.setup()
