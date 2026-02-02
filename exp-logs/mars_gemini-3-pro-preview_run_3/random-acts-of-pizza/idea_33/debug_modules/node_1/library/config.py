import os


class Config:
    """
    Global configuration for the Enhanced-Text Pent-View Stacking Ensemble.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    WORKING_DIR = "./working/idea_33"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42
    N_FOLDS = 5

    # =========================================================================
    # Data Definitions
    # =========================================================================
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    # Text columns to be concatenated (Title + Body)
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Behavioral column (List of subreddits)
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # Allow-listed Dense Metadata Features
    # Includes Raw Timestamp and User Stats.
    # Excludes _at_retrieval columns and derived text lengths.
    DENSE_FEATURES = [
        "unix_timestamp_of_request_utc",
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
    ]

    # =========================================================================
    # Feature Engineering Hyperparameters
    # =========================================================================

    # Behavioral View Constraint (Top K subreddits)
    MAX_SUBREDDIT_VOCAB = 1000

    # Lexical View (TF-IDF)
    TFIDF_PARAMS = {
        "strip_accents": "unicode",
        "stop_words": "english",
        "min_df": 5,
        "sublinear_tf": True,
        "ngram_range": (1, 2),
        "max_features": 10000,
    }

    # Semantic View (Embeddings)
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # =========================================================================
    # Model Hyperparameters (Level 1 Base Learners)
    # =========================================================================

    # 1. Sparse Lexical Branch (Enhanced Lexical Bagger - RF)
    # Regularized with min_samples_leaf=2
    RF_LEXICAL_PARAMS = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1,
    }

    # 2. Sparse Behavioral Branch (Constrained Community Bagger - RF)
    # Operates on constrained vocabulary
    RF_BEHAVIORAL_PARAMS = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1,
    }

    # 3. Dense Semantic Branch (Semantic Booster - XGBoost)
    # High capacity with early stopping and scale_pos_weight
    XGB_SEMANTIC_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": SEED,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    XGB_EARLY_STOPPING_ROUNDS = 50

    # 4. Dense Semantic Branch (Semantic Bagger - RF)
    # Modality-Specific Regularization (max_depth=12, min_samples_leaf=4)
    RF_SEMANTIC_PARAMS = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1,
    }

    # 5. Contextual Branch (Metadata Anchor - Logistic Regression)
    # High-bias regularizer
    LR_CONTEXTUAL_PARAMS = {
        "C": 0.1,
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": SEED,
    }

    # =========================================================================
    # Model Hyperparameters (Level 2 Meta-Learner)
    # =========================================================================

    META_LEARNER_PARAMS = {"C": 1.0, "solver": "lbfgs", "random_state": SEED}

    @classmethod
    def setup(cls):
        """Ensures necessary working directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
