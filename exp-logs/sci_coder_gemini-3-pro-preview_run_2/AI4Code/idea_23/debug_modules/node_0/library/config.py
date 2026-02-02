import os


class Config:
    """
    Configuration for Stacked Hybrid Ranking with Semantic Anchor Content Injection.
    """

    # --------------------------------------------------------------------------
    # Directory & File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Specific working directory for this idea to ensure caching isolation
    WORKING_DIR = "./working/idea_23"
    SUBMISSION_DIR = "./submission"

    # Metadata Pointers
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths for Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # Text Processing (TF-IDF)
    # --------------------------------------------------------------------------
    # High-dimensional sparse features for Stage 1
    TFIDF_VOCAB_SIZE = 60000
    TFIDF_NGRAM_RANGE = (1, 2)
    TFIDF_MIN_DF = 2
    # Use sublinear tf scaling to dampen the effect of very frequent terms
    TFIDF_SUBLINEAR_TF = True

    # --------------------------------------------------------------------------
    # Latent Semantic Analysis (SVD)
    # --------------------------------------------------------------------------
    # Dimensionality reduction for semantic matching and content injection
    SVD_N_COMPONENTS = 128
    SVD_RANDOM_STATE = 42

    # --------------------------------------------------------------------------
    # Feature Engineering Parameters
    # --------------------------------------------------------------------------
    # Number of nearest code neighbors to consider for calculating positional stats
    # (Mean Rank, Std Dev)
    TOP_K_POS_ANCHORS = 5

    # Innovation: Semantic Anchor Content Injection
    # We inject the top N components of the *single* nearest code neighbor's
    # SVD vector into the Stage 2 model.
    ANCHOR_CONTENT_DIMS = 16

    # --------------------------------------------------------------------------
    # Model Hyperparameters: Stage 1 (Ridge Regression)
    # --------------------------------------------------------------------------
    # High-bias baseline model
    RIDGE_ALPHA = 1.0
    RIDGE_SOLVER = "auto"

    # --------------------------------------------------------------------------
    # Model Hyperparameters: Stage 2 (LightGBM)
    # --------------------------------------------------------------------------
    # Content-aware non-linear refiner
    LGBM_PARAMS = {
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "objective": "regression",
        "metric": "mae",
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }

    # Training control
    LGBM_EARLY_STOPPING_ROUNDS = 50
    LGBM_VERBOSE_EVAL = 100  # Print metrics every 100 rounds

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
