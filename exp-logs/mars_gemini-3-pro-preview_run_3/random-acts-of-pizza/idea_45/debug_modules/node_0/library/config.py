import os


class Config:
    # =========================================================================
    # PATHS
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for caching processed features and models
    CACHE_DIR = "./working/idea_45"
    SUBMISSION_DIR = "./submission"

    # Ensure cache and submission directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # DATA CONFIGURATION
    # =========================================================================
    # Columns
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"
    TEXT_COL = "request_text_edit_aware"  # Use edit-aware text to prevent leakage
    TITLE_COL = "request_title"
    SUBREDDIT_LIST_COL = "requester_subreddits_at_request"

    # Metadata columns to allow-list (Positive Feature Selection)
    # Strictly excluding _at_retrieval columns to prevent leakage.
    # Includes Raw Timestamp for temporal anchoring.
    METADATA_COLS = [
        "unix_timestamp_of_request_utc",
        "requester_account_age_in_days_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_plus_downvotes_at_request",
    ]

    # =========================================================================
    # GENERAL SETTINGS
    # =========================================================================
    SEED = 42
    N_FOLDS = 5

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500  # Number of samples to use if DEBUG is True

    # =========================================================================
    # FEATURE ENGINEERING PARAMETERS
    # =========================================================================
    # 1. Sparse Lexical Features (Text)
    LEXICAL_TFIDF_PARAMS = {
        "max_features": 5000,
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "min_df": 5,
        "stop_words": "english",
        "strip_accents": "unicode",
    }

    # 2. Sparse Behavioral Features (Subreddit History)
    # Treated as Bag-of-Concepts, limited vocabulary to avoid overfitting rare communities
    COMMUNITY_TFIDF_PARAMS = {
        "max_features": 1000,
        "ngram_range": (1, 1),
        "sublinear_tf": True,
        "min_df": 2,
        "preprocessor": lambda x: x,  # Identity, as input is already list of strings
    }

    # 3. Dense Semantic Features (Embeddings)
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    # =========================================================================
    # MODEL HYPERPARAMETERS (LEVEL 1: BASE LEARNERS)
    # =========================================================================

    # --- Branch 1: Sparse Lexical (Text Modality) ---
    # Lexical Bagger: Random Forest on TF-IDF + Metadata
    LEXICAL_RF_PARAMS = {
        "n_estimators": 500,
        "min_samples_leaf": 2,  # Regularization
        "max_depth": None,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
    }

    # --- Branch 2: Sparse Behavioral (History Modality) ---
    # Community Bagger: Random Forest on Subreddit History + Metadata
    COMMUNITY_RF_PARAMS = {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
    }

    # --- Branch 3: Dense Semantic (Text Modality) ---
    # Semantic Booster: XGBoost on Embeddings + Metadata
    SEMANTIC_XGB_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": "cuda",  # Use GPU
        "n_jobs": -1,
        "random_state": SEED,
        "enable_categorical": False,
        "eval_metric": "auc",
        # scale_pos_weight will be set dynamically during training
    }

    # Semantic Gradient: LightGBM on Embeddings + Metadata (Algorithmic Diversity)
    SEMANTIC_LGBM_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "device": "gpu",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": -1,
        "metric": "auc",
    }

    # Semantic Bagger: Random Forest on Embeddings + Metadata (Structural Diversity)
    SEMANTIC_RF_PARAMS = {
        "n_estimators": 500,
        "max_depth": 12,  # Restricted depth for dense features
        "min_samples_leaf": 4,  # Higher leaf regularization
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
    }

    # --- Branch 4: Contextual (Metadata Modality) ---
    # Metadata Anchor: Linear Model (High Bias Regularizer)
    METADATA_LR_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": SEED,
    }

    # Temporal Booster: Non-Linear Tree on Metadata (Captures Temporal Drift)
    TEMPORAL_LGBM_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 15,  # Small trees for low-dim data
        "subsample": 0.8,
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": -1,
        "metric": "auc",
    }

    # =========================================================================
    # MODEL HYPERPARAMETERS (LEVEL 2: META LEARNER)
    # =========================================================================
    META_LR_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "random_state": SEED,
    }
