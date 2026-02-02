import os


class Config:
    """
    Configuration for the Leakage-Robust Hex-View Stacking Ensemble.
    Centralizes paths, feature definitions, and model hyperparameters.
    """

    # -------------------------------------------------------------------------
    # Directories and Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_49"
    SUBMISSION_DIR = "./submission"
    MODEL_DIR = os.path.join(WORKING_DIR, "models")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Data File Paths (using Parquet metadata)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Global Settings
    # -------------------------------------------------------------------------
    SEED = 42
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    # Text Columns: Use Edit-Aware text to prevent leakage
    TEXT_COL = "request_text_edit_aware"
    TITLE_COL = "request_title"

    # Subreddit History Column
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # -------------------------------------------------------------------------
    # Feature Selection (Allow-List)
    # -------------------------------------------------------------------------
    # Strictly allow-listed features to prevent leakage from retrieval-time stats.
    # Includes:
    # 1. User Stats (Account age, karma, activity)
    # 2. Raw Timestamp (For temporal regime discovery)
    # 3. Restored RAOP History (Valid pre-request signals)
    DENSE_FEATURES = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",  # Restored
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",  # Restored
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",  # Restored
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        "unix_timestamp_of_request_utc",  # Raw Timestamp
    ]

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------

    # 1. Lexical Bagger (Random Forest)
    # Trained on TF-IDF of Title + Body
    PARAMS_LEXICAL_BAGGER = {
        "n_estimators": 500,
        "min_samples_leaf": 2,  # Regularization
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": SEED,
        "class_weight": "balanced",
    }

    PARAMS_TFIDF_LEXICAL = {
        "ngram_range": (1, 2),
        "min_df": 5,
        "sublinear_tf": True,
        "stop_words": "english",
        "max_features": 20000,
    }

    # 2. Community Bagger (Random Forest)
    # Trained on Bag-of-Subreddits
    PARAMS_COMMUNITY_BAGGER = {
        "n_estimators": 500,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": SEED,
        "class_weight": "balanced",
    }

    PARAMS_TFIDF_COMMUNITY = {
        "max_features": 1000,  # Strict vocabulary limit
        "binary": True,
        "use_idf": False,
        "norm": None,
    }

    # 3. Semantic Booster (XGBoost)
    # Trained on Dense Embeddings + Metadata
    PARAMS_SEMANTIC_BOOSTER = {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 3.0,
        "n_jobs": -1,
        "random_state": SEED,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
        "eval_metric": "auc",
    }

    # 4. Semantic Bagger (Random Forest)
    # Structural diversity for dense embeddings
    PARAMS_SEMANTIC_BAGGER = {
        "n_estimators": 500,
        "max_depth": 12,  # Modality-specific regularization
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": SEED,
        "class_weight": "balanced",
    }

    # 5. Temporal Booster (LightGBM)
    # Trained on Metadata (can split on Raw Timestamp)
    PARAMS_TEMPORAL_BOOSTER = {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 3.0,
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": -1,
        "metric": "auc",
    }

    # 6. Metadata Anchor (Logistic Regression)
    # High-bias regularizer for metadata
    PARAMS_METADATA_ANCHOR = {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": SEED,
    }

    # Level 2 Meta-Learner (Logistic Regression)
    PARAMS_META_LEARNER = {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "random_state": SEED,
    }

    # Embeddings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
