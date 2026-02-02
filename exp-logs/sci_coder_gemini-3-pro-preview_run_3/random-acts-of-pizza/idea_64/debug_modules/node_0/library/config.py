import os


class Config:
    """
    Central Configuration for the Pizza Request Success Prediction Task.
    Implements the High-Fidelity Hept-View Stacking Ensemble architecture settings.
    """

    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42
    N_FOLDS = 5
    TARGET_COL = "requester_received_pizza"
    ID_COL = "request_id"

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_64"
    SUBMISSION_DIR = "./submission"

    # Input Data Paths (Metadata Parquet Files)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Feature Engineering Configuration
    # =========================================================================

    # Text Columns for Concatenation
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Community/Behavioral Column
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # Metadata Allow-List (Strict Feature Selection)
    # Excludes retrieval-time features to prevent leakage
    METADATA_COLS = [
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
    # Model Hyperparameters (Level 1 Base Learners)
    # =========================================================================

    # --- Branch 1: Sparse Lexical (Text Modality) ---
    # High-Fidelity Lexical Bagger (Random Forest)
    # Captures granular signals (e.g., "I", "$") and the long tail
    LEXICAL_RF_PARAMS = {
        "n_estimators": 200,
        "min_samples_leaf": 2,  # Regularization
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
        "class_weight": "balanced",
    }

    LEXICAL_VECTORIZER_PARAMS = {
        "token_pattern": r"\w{1,}",  # Capture single characters
        "min_df": 2,  # Preserve long tail
        "sublinear_tf": True,
        "stop_words": "english",
        "ngram_range": (1, 2),
    }

    # --- Branch 2: Sparse Behavioral (History Modality) ---
    # Community Bagger (Random Forest)
    # Bag-of-Concepts approach for subreddits
    COMMUNITY_RF_PARAMS = {
        "n_estimators": 200,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
        "class_weight": "balanced",
    }

    COMMUNITY_VECTORIZER_PARAMS = {
        "max_features": 1000,  # Constrained vocabulary for communities
        "binary": False,  # Use TF-IDF weighting
        "sublinear_tf": True,
        "stop_words": "english",
    }

    # --- Branch 3: Dense Semantic (Text Modality) ---
    # Embedding Model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Semantic Booster (XGBoost)
    # Conservative boosting to reduce variance
    SEMANTIC_XGB_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "n_jobs": -1,
        "random_state": SEED,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "scale_pos_weight": 3.0,  # Approx ratio (75/25)
    }

    # Semantic Gradient (LightGBM)
    # Algorithmic diversity (Leaf-wise)
    SEMANTIC_LGBM_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "n_jobs": -1,
        "random_state": SEED,
        "objective": "binary",
        "metric": "auc",
        "verbose": -1,
        "class_weight": "balanced",
    }

    # Semantic Bagger (Random Forest)
    # Structural diversity with modality-specific regularization
    SEMANTIC_RF_PARAMS = {
        "n_estimators": 200,
        "max_depth": 12,  # Prevent memorization of dense noise
        "min_samples_leaf": 4,
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
        "class_weight": "balanced",
    }

    # --- Branch 4: Contextual (Metadata Modality) ---
    # Metadata Anchor (Logistic Regression)
    # High-bias regularizer
    METADATA_ANCHOR_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "random_state": SEED,
        "max_iter": 1000,
        "class_weight": "balanced",
    }

    # Temporal Booster (LightGBM)
    # Captures non-linear temporal drift
    METADATA_BOOSTER_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 15,  # Restricted complexity for fewer features
        "n_jobs": -1,
        "random_state": SEED,
        "objective": "binary",
        "metric": "auc",
        "verbose": -1,
        "class_weight": "balanced",
    }

    # =========================================================================
    # Level 2 Meta-Learner
    # =========================================================================
    META_LEARNER_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "random_state": SEED,
    }

    # =========================================================================
    # Training Configuration
    # =========================================================================
    EARLY_STOPPING_ROUNDS = 100

    @classmethod
    def setup(cls):
        """
        Creates necessary working and submission directories.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
