import os


class Config:
    # =========================================
    # Global Configuration
    # =========================================
    SEED = 42
    N_FOLDS = 5  # Number of folds for Stacking (Level 1 OOF generation)
    INTERNAL_VAL_SIZE = (
        0.1  # Size of internal validation set for XGBoost early stopping
    )

    # =========================================
    # Directories and Paths
    # =========================================
    # Input Metadata (Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working Directory for Caching (Write Allowed)
    # Stores intermediate processed features (embeddings, TF-IDF matrices)
    CACHE_DIR = "./working/idea_28"

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================
    # Data Columns
    # =========================================
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    # Use edit-aware text to prevent leakage from "EDIT: Thanks for pizza"
    TEXT_COL = "request_text_edit_aware"
    TITLE_COL = "request_title"

    # Subreddit history column (list of strings)
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # Metadata Allow-List:
    # Only features strictly available at the time of request.
    # Excludes all *_at_retrieval columns and derived text stats.
    NUMERICAL_COLS = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        "unix_timestamp_of_request",
    ]

    # =========================================
    # Feature Engineering Parameters
    # =========================================
    # Dense Embeddings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Sparse TF-IDF
    TFIDF_MAX_FEATURES = 3000
    TFIDF_PARAMS = {
        "sublinear_tf": True,
        "min_df": 5,
        "stop_words": "english",
        "ngram_range": (1, 2),
    }

    # =========================================
    # Model Hyperparameters
    # =========================================

    # 1. Random Forest (Lexical Bagger, Community Bagger, Semantic Bagger)
    # Regularized via min_samples_leaf to prevent overfitting on sparse data
    RF_PARAMS = {
        "n_estimators": 500,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
    }

    # 2. XGBoost (Semantic Booster)
    # High capacity model, controlled via Nested Internal Validation
    # Configured for GPU (A100)
    XGB_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": "cuda",
        "n_jobs": -1,
        "random_state": SEED,
        "enable_categorical": False,
        "verbosity": 0,
    }
    XGB_EARLY_STOPPING_ROUNDS = 50

    # 3. Logistic Regression (Metadata Anchor, Meta-Learner)
    # High bias regularizer
    LR_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": SEED,
        "n_jobs": -1,
    }
