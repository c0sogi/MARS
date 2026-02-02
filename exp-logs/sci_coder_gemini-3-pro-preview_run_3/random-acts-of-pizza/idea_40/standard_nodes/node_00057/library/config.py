import os


class Config:
    # =========================================================================
    # Global Configuration
    # =========================================================================
    RANDOM_SEED = 42
    N_FOLDS = 5

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Metadata (Pre-split)
    TRAIN_METADATA_PATH = "./metadata/train.parquet"
    VAL_METADATA_PATH = "./metadata/val.parquet"
    TEST_METADATA_PATH = "./metadata/test.parquet"

    # Working Directory for Caching
    # We use idea_40 as the identifier for this iteration
    CACHE_DIR = "./working/idea_40/"

    # Submission
    SUBMISSION_DIR = "./submission/"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Feature Engineering Configuration
    # =========================================================================
    # Text Processing
    TEXT_COLS = ["request_title", "request_text_edit_aware"]
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

    # TF-IDF Settings
    TFIDF_PARAMS = {
        "sublinear_tf": True,
        "min_df": 5,
        "stop_words": "english",
        "ngram_range": (1, 2),
        "max_features": 10000,  # Cap to prevent explosion, though RF handles sparse well
    }

    # Behavioral/Community Processing
    SUBREDDIT_COL = "requester_subreddits_at_request"
    TOP_K_SUBREDDITS = 1000  # Limit vocabulary for community bagger

    # Latent Interaction (SVD)
    SVD_COMPONENTS_TEXT = 32
    SVD_COMPONENTS_HISTORY = 32

    # Metadata (Positive Feature Selection - Allow List)
    # Explicitly excluding retrieval-time features and derived text lengths
    METADATA_DENSE_COLS = [
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
    # Model Hyperparameters (Level 1)
    # =========================================================================

    # 1. Lexical Bagger (Random Forest on Sparse Text)
    RF_LEXICAL_PARAMS = {
        "n_estimators": 300,
        "min_samples_leaf": 2,  # Regularization
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbose": 0,
    }

    # 2. Community Bagger (Random Forest on Sparse History)
    RF_COMMUNITY_PARAMS = {
        "n_estimators": 300,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbose": 0,
    }

    # 3. Semantic Booster (XGBoost on Dense Embeddings)
    XGB_SEMANTIC_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": "cuda",  # Use GPU
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "early_stopping_rounds": 50,
        "verbosity": 0,
        # scale_pos_weight will be set dynamically during training
    }

    # 4. Semantic Bagger (Random Forest on Dense Embeddings)
    RF_SEMANTIC_PARAMS = {
        "n_estimators": 300,
        "max_depth": 12,  # Modality-specific regularization
        "min_samples_leaf": 4,  # Modality-specific regularization
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbose": 0,
    }

    # 5. Interaction Booster (LightGBM on SVD Text + SVD History)
    LGBM_INTERACTION_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "num_leaves": 31,
        "device": "gpu",  # Use GPU
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "early_stopping_rounds": 50,
        # scale_pos_weight/is_unbalance will be handled dynamically
    }

    # 6. Metadata Anchor (Logistic Regression)
    LR_ANCHOR_PARAMS = {
        "penalty": "l2",
        "C": 1.0,
        "class_weight": "balanced",
        "solver": "liblinear",
        "random_state": RANDOM_SEED,
        "max_iter": 1000,
    }

    # =========================================================================
    # Meta-Learner Configuration (Level 2)
    # =========================================================================
    META_LEARNER_PARAMS = {
        "penalty": "l2",
        "C": 0.5,  # Stronger regularization for meta-learner
        "class_weight": None,  # Let the meta-learner calibrate probabilities naturally
        "solver": "liblinear",
        "random_state": RANDOM_SEED,
    }

    @classmethod
    def ensure_directories(cls):
        """Creates necessary directories for cache and submission."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
