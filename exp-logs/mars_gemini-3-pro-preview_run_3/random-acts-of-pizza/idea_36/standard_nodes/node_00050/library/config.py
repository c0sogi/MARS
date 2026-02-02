import os


class Config:
    """
    Configuration for the Pent-View Stacking Ensemble with Bayesian Community Profiling.
    """

    # ==========================================
    # PATHS & DIRECTORIES
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_36"
    SUBMISSION_DIR = "./submission"

    # Input Metadata Files (Parquet)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output Submission File
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # GLOBAL SETTINGS
    # ==========================================
    SEED = 42
    N_FOLDS = 5
    TARGET_COL = "requester_received_pizza"
    ID_COL = "request_id"

    # Debugging: Set to an integer (e.g., 100) to subsample data for rapid testing.
    # Set to None for full training.
    DEBUG_SAMPLE_SIZE = None

    # ==========================================
    # FEATURE ENGINEERING CONFIGURATION
    # ==========================================

    # --- Text Modality ---
    # Columns to concatenate for text analysis
    # Note: 'request_text' is used in train, 'request_text_edit_aware' in test (to avoid leakage)
    TRAIN_TEXT_COLS = ["request_title", "request_text"]
    TEST_TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Sparse Lexical Features (TF-IDF)
    TFIDF_PARAMS = {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 5,
        "sublinear_tf": True,
        "stop_words": "english",
        "strip_accents": "unicode",
    }

    # Dense Semantic Features (Embeddings)
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    # --- Behavioral Modality (Community History) ---
    COMMUNITY_COL = "requester_subreddits_at_request"
    # Limit vocabulary to top K subreddits to prevent overfitting to rare communities
    COMMUNITY_VOCAB_SIZE = 1000

    # --- Contextual Modality (Metadata) ---
    # Allow-list for numerical metadata features
    METADATA_DENSE_FEATURES = [
        "unix_timestamp_of_request_utc",
        "requester_account_age_in_days_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
    ]

    # Name for the engineered Bayesian Community Success Score
    COMMUNITY_SCORE_COL = "community_success_score"

    # ==========================================
    # MODEL HYPERPARAMETERS
    # ==========================================

    # 1. Sparse Lexical Branch: Lexical Bagger (Random Forest)
    # Trained on TF-IDF (Title+Body) + Metadata
    MODEL_LEXICAL_RF = {
        "n_estimators": 300,
        "min_samples_leaf": 2,  # Regularization
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": 0,
    }

    # 2. Sparse Behavioral Branch: Community Bagger (Random Forest)
    # Trained on TF-IDF (Subreddit History) + Metadata
    MODEL_COMMUNITY_RF = {
        "n_estimators": 300,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": 0,
    }

    # 3. Dense Semantic Branch: Semantic Booster (XGBoost)
    # Trained on Embeddings + Metadata + Community Score
    MODEL_SEMANTIC_XGB = {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 3.0,  # Handle class imbalance (~1:3)
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "early_stopping_rounds": 100,
    }

    # 4. Dense Semantic Branch: Semantic Bagger (Random Forest)
    # Trained on Embeddings + Metadata
    MODEL_SEMANTIC_RF = {
        "n_estimators": 300,
        "max_depth": 12,  # Modality-specific regularization
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": 0,
    }

    # 5. Contextual Branch: Metadata Anchor (Logistic Regression)
    # Trained on Metadata only
    MODEL_METADATA_LR = {
        "C": 0.1,  # Strong regularization for high bias
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": SEED,
    }

    # Level 2 Meta-Learner (Logistic Regression)
    MODEL_META_LEARNER = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "random_state": SEED,
    }

    @classmethod
    def create_dirs(cls):
        """Creates necessary working and submission directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
