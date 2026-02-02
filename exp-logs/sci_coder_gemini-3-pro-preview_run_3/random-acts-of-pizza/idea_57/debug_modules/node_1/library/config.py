import os


class Config:
    # =========================================================================
    # PATHS & DIRECTORIES
    # =========================================================================
    # Input Metadata Paths
    TRAIN_PATH = "./metadata/train.parquet"
    VAL_PATH = "./metadata/val.parquet"
    TEST_PATH = "./metadata/test.parquet"

    # Working Directories
    WORKING_DIR = "./working/idea_57/"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_DIR = os.path.join(WORKING_DIR, "models")
    SUBMISSION_DIR = "./submission/"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # GLOBAL SETTINGS
    # =========================================================================
    RANDOM_STATE = 42
    N_FOLDS = 5
    N_JOBS = 12  # Utilizing available vCPUs

    # Early Stopping for Volatile Learners (XGB/LGBM)
    EARLY_STOPPING_ROUNDS = 50

    # =========================================================================
    # DATA DEFINITIONS
    # =========================================================================
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    # Text Features
    # We use edit_aware to prevent leakage from edits saying "Thanks for pizza"
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Behavioral/Community Features
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # Metadata Allow-List
    # Strictly excluding retrieval-time features to prevent leakage.
    # Including raw timestamp and RAOP history as per "Lessons Learned".
    METADATA_COLS = [
        "unix_timestamp_of_request_utc",
        "requester_account_age_in_days_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_on_raop_at_request",
    ]

    # =========================================================================
    # FEATURE ENGINEERING CONFIG
    # =========================================================================
    # Sparse Lexical Features (TF-IDF)
    TEXT_TFIDF_PARAMS = {
        "min_df": 5,
        "max_features": 20000,
        "sublinear_tf": True,
        "ngram_range": (1, 2),
        "stop_words": "english",
        "strip_accents": "unicode",
    }

    # Sparse Community Features (Bag of Concepts)
    # Limited to Top 1000 to prevent overfitting and treat subreddits as concepts
    SUBREDDIT_TFIDF_PARAMS = {
        "min_df": 2,
        "max_features": 1000,
        "binary": True,
        "stop_words": None,
        "token_pattern": r"(?u)\b\w+\b",  # Simple tokenization for subreddit names
    }

    # Dense Semantic Features
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    # =========================================================================
    # MODEL HYPERPARAMETERS (LEVEL 1)
    # =========================================================================

    # --- Branch 1: Sparse Lexical (Text) ---
    # Lexical Bagger: Random Forest on Text
    LEXICAL_BAGGER_PARAMS = {
        "n_estimators": 200,
        "min_samples_leaf": 2,  # Regularization
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
    }

    # Lexical Anchor: Logistic Regression on Text
    LEXICAL_ANCHOR_PARAMS = {
        "C": 1.0,
        "solver": "liblinear",
        "penalty": "l2",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
    }

    # --- Branch 2: Sparse Behavioral (History) ---
    # Community Bagger: Random Forest on Subreddits
    COMMUNITY_BAGGER_PARAMS = {
        "n_estimators": 200,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
    }

    # Community Anchor: Logistic Regression on Subreddits
    COMMUNITY_ANCHOR_PARAMS = {
        "C": 1.0,
        "solver": "liblinear",
        "penalty": "l2",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
    }

    # --- Branch 3: Dense Semantic (Embeddings) ---
    # Semantic Booster: XGBoost on Embeddings
    SEMANTIC_BOOSTER_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        # scale_pos_weight will be set dynamically during training
    }

    # Semantic Gradient: LightGBM on Embeddings
    SEMANTIC_GRADIENT_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "class_weight": "balanced",
    }

    # Semantic Bagger: Random Forest on Embeddings
    # Modality-Specific Regularization applied (deeper depth, higher leaf min)
    SEMANTIC_BAGGER_PARAMS = {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
    }

    # --- Branch 4: Contextual (Metadata) ---
    # Metadata Anchor: Logistic Regression on Metadata
    # High-bias regularizer (C=0.1)
    METADATA_ANCHOR_PARAMS = {
        "C": 0.1,
        "solver": "liblinear",
        "penalty": "l2",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
    }

    # Temporal Booster: LightGBM on Metadata
    TEMPORAL_BOOSTER_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "num_leaves": 15,  # Lower complexity for low-dim feature space
        "verbose": -1,
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "class_weight": "balanced",
    }

    # =========================================================================
    # META LEARNER (LEVEL 2)
    # =========================================================================
    META_LEARNER_PARAMS = {
        "C": 1.0,
        "solver": "lbfgs",
        "penalty": "l2",
        "random_state": RANDOM_STATE,
        "max_iter": 1000,
    }

    # =========================================================================
    # ENSEMBLE DEFINITION
    # =========================================================================
    # Maps model keys to their configuration for the pipeline
    MODEL_CONFIGS = {
        "lexical_bagger": {
            "type": "sklearn_rf",
            "params": LEXICAL_BAGGER_PARAMS,
            "feature_set": "lexical_sparse",
        },
        "lexical_anchor": {
            "type": "sklearn_lr",
            "params": LEXICAL_ANCHOR_PARAMS,
            "feature_set": "lexical_sparse",
        },
        "community_bagger": {
            "type": "sklearn_rf",
            "params": COMMUNITY_BAGGER_PARAMS,
            "feature_set": "community_sparse",
        },
        "community_anchor": {
            "type": "sklearn_lr",
            "params": COMMUNITY_ANCHOR_PARAMS,
            "feature_set": "community_sparse",
        },
        "semantic_booster": {
            "type": "xgboost",
            "params": SEMANTIC_BOOSTER_PARAMS,
            "feature_set": "semantic_dense",
        },
        "semantic_gradient": {
            "type": "lightgbm",
            "params": SEMANTIC_GRADIENT_PARAMS,
            "feature_set": "semantic_dense",
        },
        "semantic_bagger": {
            "type": "sklearn_rf",
            "params": SEMANTIC_BAGGER_PARAMS,
            "feature_set": "semantic_dense",
        },
        "metadata_anchor": {
            "type": "sklearn_lr",
            "params": METADATA_ANCHOR_PARAMS,
            "feature_set": "metadata_only",
        },
        "temporal_booster": {
            "type": "lightgbm",
            "params": TEMPORAL_BOOSTER_PARAMS,
            "feature_set": "metadata_only",
        },
    }
