import os


class Config:
    """
    Central configuration for the Pizza Request Success Prediction task.
    Implements the Unified Interaction-Aware Stacking Ensemble architecture.
    """

    # =========================================================================
    # PATHS & DIRECTORIES
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_35"
    SUBMISSION_DIR = "./submission"

    # File Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # DATA CONFIGURATION
    # =========================================================================
    TARGET_COL = "requester_received_pizza"
    ID_COL = "request_id"

    # Text Inputs
    TEXT_COLS = ["request_title", "request_text_edit_aware"]
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # Metadata Allow-List (Positive Feature Selection)
    # Explicitly excludes _at_retrieval columns to prevent leakage
    METADATA_COLS = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        "unix_timestamp_of_request_utc",  # Raw Temporal Anchor
    ]

    # =========================================================================
    # FEATURE ENGINEERING SETTINGS
    # =========================================================================
    RANDOM_STATE = 42
    N_FOLDS = 5

    # TF-IDF Settings
    TFIDF_PARAMS = {
        "min_df": 5,
        "sublinear_tf": True,
        "ngram_range": (1, 2),
        "stop_words": "english",
        "strip_accents": "unicode",
    }

    # Vocabulary Limits for Sparse Bags
    VOCAB_UNIFIED = 5000
    VOCAB_LEXICAL = 5000
    VOCAB_COMMUNITY = 1000  # Strictly limited to top 1000 subreddits

    # Dense Embeddings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE = 32

    # =========================================================================
    # MODEL HYPERPARAMETERS
    # =========================================================================

    # --- Level 1: Base Learners ---

    # 1. Unified Sparse Branch: Interaction Bagger (RF)
    # Inputs: Title + Body + Prefixed History (TF-IDF) + Metadata
    MODEL_UNIFIED_RF = {
        "n_estimators": 300,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "verbose": 0,
    }

    # 2. Specialized Sparse Branches
    # Lexical Bagger (RF)
    # Inputs: Title + Body (TF-IDF) + Metadata
    MODEL_LEXICAL_RF = {
        "n_estimators": 300,
        "min_samples_leaf": 2,  # Regularization
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "verbose": 0,
    }

    # Community Bagger (RF)
    # Inputs: Subreddit History (TF-IDF) + Metadata
    MODEL_COMMUNITY_RF = {
        "n_estimators": 300,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "verbose": 0,
    }

    # 3. Dense Semantic Branch
    # Semantic Booster (XGBoost)
    # Inputs: Embeddings + Metadata
    MODEL_SEMANTIC_XGB = {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "eval_metric": "auc",
        "early_stopping_rounds": 50,
        "verbosity": 0,
        # scale_pos_weight is calculated dynamically in pipeline
    }

    # Semantic Bagger (RF)
    # Inputs: Embeddings + Metadata
    MODEL_SEMANTIC_RF = {
        "n_estimators": 300,
        "max_depth": 12,  # Modality-Specific Regularization
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "verbose": 0,
    }

    # 4. Contextual Branch
    # Metadata Anchor (Logistic Regression)
    # Inputs: Metadata only
    MODEL_METADATA_LR = {
        "C": 1.0,
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
    }

    # --- Level 2: Meta-Learner ---

    # Stacking Meta-Learner (Logistic Regression)
    MODEL_META_LR = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
    }
