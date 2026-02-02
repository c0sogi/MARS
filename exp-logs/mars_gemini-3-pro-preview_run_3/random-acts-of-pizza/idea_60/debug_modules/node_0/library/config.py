import os


class Config:
    """
    Configuration for the Granular Hept-View Stacking Ensemble.
    Defines paths, global constants, and hyperparameters for all 7 base learners
    and the meta-learner.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_60"  # Cache directory
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Global Settings
    # =========================================================================
    RANDOM_SEED = 42
    N_FOLDS = 5
    TARGET_COL = "requester_received_pizza"
    ID_COL = "request_id"
    DEBUG = False  # Set to True to run on a subset of data for debugging

    # =========================================================================
    # Feature Engineering & Selection
    # =========================================================================
    # Text columns to be concatenated
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Explicit Allow-List for Global Metadata (Signal Preservation)
    METADATA_COLS = [
        "unix_timestamp_of_request_utc",  # Raw timestamp for temporal drift
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",  # Restored prior
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",  # Restored prior
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
    ]

    # Columns to strictly exclude (Leakage Prevention)
    EXCLUDE_SUFFIX = "_at_retrieval"

    # =========================================================================
    # Model Hyperparameters (Level 1 Base Learners)
    # =========================================================================

    # --- Branch 1: Sparse Lexical (Text Modality) ---
    # Granular Lexical Bagger (Random Forest)
    LEXICAL_RF_PARAMS = {
        "n_estimators": 200,
        "min_samples_leaf": 2,  # Regularization
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    LEXICAL_VECTORIZER_PARAMS = {
        "strip_accents": "unicode",
        "lowercase": True,
        "analyzer": "word",
        "token_pattern": r"\w{1,}",  # Granular tokenization (captures "I", "$")
        "ngram_range": (1, 2),
        "min_df": 5,
        "sublinear_tf": True,
        "max_features": 20000,
    }

    # --- Branch 2: Sparse Behavioral (History Modality) ---
    # Community Bagger (Random Forest)
    COMMUNITY_RF_PARAMS = {
        "n_estimators": 200,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    COMMUNITY_VECTORIZER_PARAMS = {
        "max_features": 1000,  # Top 1000 subreddits only
        "binary": True,  # Bag-of-Concepts approach
        "token_pattern": r"(?u)\b\w\w+\b",  # Standard tokenization for subreddits
    }

    # --- Branch 3: Dense Semantic (Text Modality) ---
    # Shared Embedding Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Semantic Booster (XGBoost) - Conservative Boosting
    SEMANTIC_XGB_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.01,  # Conservative
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.6,  # Conservative
        "scale_pos_weight": 3.0,  # Imbalance handling (~3:1 neg:pos ratio)
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 100,
    }

    # Semantic Gradient (LightGBM) - Leaf-wise Growth
    SEMANTIC_LGBM_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "colsample_bytree": 0.7,
        "subsample": 0.8,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": -1,
        "early_stopping_rounds": 100,
    }

    # Semantic Bagger (Random Forest) - Structural Diversity
    SEMANTIC_RF_PARAMS = {
        "n_estimators": 200,
        "max_depth": 12,  # Modality-Specific Regularization
        "min_samples_leaf": 4,  # Modality-Specific Regularization
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    # --- Branch 4: Contextual (Metadata Modality) ---
    # Metadata Anchor (Logistic Regression) - High Bias Regularizer
    METADATA_LR_PARAMS = {
        "C": 0.1,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
    }

    # Temporal Booster (LightGBM) - Non-linear Temporal Drift
    METADATA_LGBM_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.01,
        "num_leaves": 15,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": -1,
        "early_stopping_rounds": 100,
    }

    # =========================================================================
    # Level 2 Meta-Learner
    # =========================================================================
    META_LEARNER_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "random_state": RANDOM_SEED,
    }
