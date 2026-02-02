import os


class Config:
    """
    Configuration for the Leakage-Robust Hex-View Stacking Ensemble.
    Defines project-wide constants, paths, feature sets, and model hyperparameters.
    """

    # -------------------------------------------------------------------------
    # 1. Directories and Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific cache directory for this idea
    CACHE_DIR = "./working/idea_46"
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths (Parquet format)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Final Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # 2. Global Settings & Debugging
    # -------------------------------------------------------------------------
    RANDOM_SEED = 42
    N_FOLDS = 5

    # Debugging flags to control dataset size
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500

    # -------------------------------------------------------------------------
    # 3. Data Definitions & Feature Selection
    # -------------------------------------------------------------------------
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    # Text Columns
    # NOTE: We strictly use 'request_text_edit_aware' to prevent target leakage
    # from post-hoc edits (e.g., "EDIT: Thanks for the pizza").
    TEXT_COL_TITLE = "request_title"
    TEXT_COL_BODY = "request_text_edit_aware"

    # Subreddit History Column (List of strings)
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # Allow-listed Metadata Features
    # Explicitly selected to include raw temporal anchors and exclude retrieval-time leakage.
    METADATA_DENSE_FEATURES = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        # Critical: Raw timestamp for capturing non-cyclic temporal drift
        "unix_timestamp_of_request_utc",
    ]

    # -------------------------------------------------------------------------
    # 4. Feature Engineering & Vectorization Parameters
    # -------------------------------------------------------------------------

    # Lexical (Text) Vectorization
    TFIDF_TEXT_PARAMS = {
        "ngram_range": (1, 2),
        "min_df": 5,
        "max_features": 10000,
        "sublinear_tf": True,
        "stop_words": "english",
        "lowercase": True,
        "strip_accents": "unicode",
    }

    # Behavioral (Subreddit) Vectorization
    # Treated as a Bag-of-Concepts, limited vocab to prevent overfitting
    TFIDF_SUBREDDIT_PARAMS = {
        "ngram_range": (1, 1),
        "min_df": 2,
        "max_features": 1000,
        "binary": True,
        "lowercase": False,
        "preprocessor": lambda x: x,  # Identity, assumes input is joined string or handled by custom analyzer
    }

    # Semantic (Embedding) Configuration
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    EMBEDDING_BATCH_SIZE = 32

    # -------------------------------------------------------------------------
    # 5. Level 1 Base Learner Hyperparameters
    # -------------------------------------------------------------------------

    # 1. Lexical Bagger: Random Forest on Text TF-IDF + Metadata
    L1_LEXICAL_RF_PARAMS = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbose": 0,
    }

    # 2. Community Bagger: Random Forest on Subreddit TF-IDF + Metadata
    L1_COMMUNITY_RF_PARAMS = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbose": 0,
    }

    # 3. Semantic Booster: XGBoost on Embeddings + Metadata
    # Uses scale_pos_weight (~3.0) for imbalance correction
    L1_SEMANTIC_XGB_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 3.0,
        "n_jobs": 12,
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
        "verbosity": 0,
    }

    # 4. Semantic Bagger: Random Forest on Embeddings + Metadata
    # Deep trees with leaf regularization
    L1_SEMANTIC_RF_PARAMS = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbose": 0,
    }

    # 5. Metadata Anchor: Logistic Regression on Metadata
    # High-bias linear baseline
    L1_META_LR_PARAMS = {
        "C": 0.1,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
    }

    # 6. Temporal Booster: LightGBM on Metadata
    # Captures non-linear temporal regimes via raw timestamp splits
    L1_META_LGBM_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "class_weight": "balanced",
        "n_jobs": 12,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "metric": "auc",
    }

    # -------------------------------------------------------------------------
    # 6. Level 2 Meta-Learner Hyperparameters
    # -------------------------------------------------------------------------
    L2_LOGREG_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "random_state": RANDOM_SEED,
    }
