import os


class Config:
    # =========================================================================
    # Global Configuration
    # =========================================================================
    SEED = 42
    N_FOLDS = 5
    N_JOBS = 12  # Utilizing available vCPUs

    # =========================================================================
    # Directory Paths
    # =========================================================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_63"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input File Paths (Metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (for deterministic processing)
    CACHE_PROCESSED_DATA_PATH = os.path.join(WORKING_DIR, "processed_data.parquet")
    CACHE_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "embeddings.npy")

    # =========================================================================
    # Data Schema & Feature Selection
    # =========================================================================
    TARGET_COL = "requester_received_pizza"
    ID_COL = "request_id"

    # Text Columns for Concatenation
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Subreddit History Column
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # Strict Allow-List for Metadata (Hygiene & Leakage Prevention)
    # Includes restored RAOP history and raw timestamps as per solution design
    METADATA_ALLOW_LIST = [
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
    # Vectorization & Embedding Configuration
    # =========================================================================
    # Granular Tokenization to capture agency ("I") and symbols ("$")
    TOKEN_PATTERN = r"\w{1,}"

    # Sparse Lexical Config
    TFIDF_TEXT_PARAMS = {
        "strip_accents": "unicode",
        "lowercase": True,
        "analyzer": "word",
        "ngram_range": (1, 2),
        "min_df": 5,
        "max_features": 5000,
        "sublinear_tf": True,
        "token_pattern": TOKEN_PATTERN,
    }

    # Sparse Behavioral Config (Community Bag-of-Concepts)
    TFIDF_COMMUNITY_PARAMS = {
        "strip_accents": "unicode",
        "lowercase": False,  # Subreddits are case sensitive or standard
        "analyzer": "word",
        "token_pattern": r"(?u)\b\w\w+\b",  # Standard pattern sufficient for subreddits
        "min_df": 2,
        "max_features": 1000,
        "binary": True,  # Presence/Absence is more robust than frequency for history
    }

    # Dense Embedding Config
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    BATCH_SIZE = 32

    # =========================================================================
    # Model Hyperparameters (Level 1: Base Learners)
    # =========================================================================

    # 1. Lexical Bagger (Sparse Text -> RF)
    LEXICAL_BAGGER_PARAMS = {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 2,  # Regularization
        "max_features": "sqrt",
        "class_weight": "balanced",
        "n_jobs": N_JOBS,
        "random_state": SEED,
        "verbose": 0,
    }

    # 2. Community Bagger (Sparse History -> RF)
    COMMUNITY_BAGGER_PARAMS = {
        "n_estimators": 500,
        "max_depth": 20,  # Prevent overfitting to rare communities
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "n_jobs": N_JOBS,
        "random_state": SEED,
        "verbose": 0,
    }

    # 3. Semantic Booster (Dense Text -> XGB)
    # Conservative settings to reduce variance
    SEMANTIC_BOOSTER_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_jobs": N_JOBS,
        "random_state": SEED,
        "tree_method": "hist",
        "early_stopping_rounds": 100,
    }

    # 4. Semantic Gradient (Dense Text -> LGBM)
    SEMANTIC_GRADIENT_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "objective": "binary",
        "metric": "auc",
        "n_jobs": N_JOBS,
        "random_state": SEED,
        "verbose": -1,
        "early_stopping_rounds": 100,
    }

    # 5. Semantic Bagger (Dense Text -> RF)
    # Modality-specific regularization
    SEMANTIC_BAGGER_PARAMS = {
        "n_estimators": 500,
        "max_depth": 12,  # Constrained depth for dense features
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "n_jobs": N_JOBS,
        "random_state": SEED,
        "verbose": 0,
    }

    # 6. Metadata Anchor (Metadata -> Logistic Regression)
    METADATA_ANCHOR_PARAMS = {
        "penalty": "l2",
        "C": 0.1,  # Stronger regularization
        "solver": "lbfgs",
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": SEED,
        "n_jobs": N_JOBS,
    }

    # 7. Temporal Booster (Metadata -> LGBM)
    # Captures non-linear time/stat interactions
    TEMPORAL_BOOSTER_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.02,
        "num_leaves": 15,  # Low complexity
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary",
        "metric": "auc",
        "n_jobs": N_JOBS,
        "random_state": SEED,
        "verbose": -1,
        "early_stopping_rounds": 100,
    }

    # =========================================================================
    # Model Hyperparameters (Level 2: Meta-Learner)
    # =========================================================================
    META_LEARNER_PARAMS = {
        "penalty": "l2",
        "C": 0.5,  # Calibrated regularization
        "solver": "lbfgs",
        "class_weight": None,  # Let the probabilities speak
        "max_iter": 1000,
        "random_state": SEED,
        "n_jobs": N_JOBS,
    }
