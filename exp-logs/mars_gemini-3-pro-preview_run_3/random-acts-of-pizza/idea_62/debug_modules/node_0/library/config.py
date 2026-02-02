import os


class Config:
    """
    Configuration for the Conservative Granular Hept-View Stacking Ensemble.
    Defines paths, constants, feature lists, and model hyperparameters.
    """

    # =========================================================================
    # Global Settings
    # =========================================================================
    RANDOM_STATE = 42
    N_FOLDS = 5
    EARLY_STOPPING_ROUNDS = 50

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Input Metadata (Pre-generated)
    INPUT_DIR = "./metadata"
    TRAIN_PATH = os.path.join(INPUT_DIR, "train.parquet")
    VAL_PATH = os.path.join(INPUT_DIR, "val.parquet")
    TEST_PATH = os.path.join(INPUT_DIR, "test.parquet")

    # Working Directory for Cache and Models
    # Using 'idea_62' as the specific iteration identifier
    WORKING_DIR = "./working/idea_62"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_DIR = os.path.join(WORKING_DIR, "models")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary writeable directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Definitions
    # =========================================================================
    TARGET_COL = "requester_received_pizza"
    ID_COL = "request_id"

    # Text Inputs
    # We use the edit-aware text to avoid leakage from "EDIT: Thanks for pizza"
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Behavioral Input
    COMMUNITY_COL = "requester_subreddits_at_request"

    # Metadata Allow-List
    # Strictly excludes columns ending in '_at_retrieval' to prevent leakage.
    # Includes restored RAOP history priors.
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
    # Feature Engineering Hyperparameters
    # =========================================================================
    # Granular token pattern to capture single characters (e.g., 'I', '$')
    TOKEN_PATTERN = r"\w{1,}"

    # Vocabulary limit for subreddit history (Bag-of-Concepts)
    VOCAB_SIZE_COMMUNITY = 1000

    # Dense Embedding Model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # =========================================================================
    # Model Hyperparameters (Level 1 Base Learners)
    # =========================================================================

    # 1. Sparse Lexical Branch: Granular Lexical Bagger
    # Random Forest on TF-IDF (Title + Body)
    LEXICAL_BAGGER_PARAMS = {
        "n_estimators": 100,
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    # 2. Sparse Behavioral Branch: Community Bagger
    # Random Forest on TF-IDF (Subreddits)
    COMMUNITY_BAGGER_PARAMS = {
        "n_estimators": 100,
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    # 3. Dense Semantic Branch

    # Semantic Booster (XGBoost) - Conservative Configuration
    # Low learning rate and strict regularization to prevent overfitting on embeddings
    SEMANTIC_BOOSTER_PARAMS = {
        "n_estimators": 2000,  # High ceiling, controlled by early stopping
        "learning_rate": 0.01,
        "max_depth": 6,
        "colsample_bytree": 0.6,
        "subsample": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        # Note: scale_pos_weight to be calculated dynamically based on training data
    }

    # Semantic Gradient (LightGBM)
    # Algorithmic diversity via leaf-wise growth
    SEMANTIC_GRADIENT_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "objective": "binary",
        "metric": "auc",
        "verbose": -1,
        # early_stopping_rounds passed to fit()
    }

    # Semantic Bagger (Random Forest)
    # Structural diversity with modality-specific regularization
    SEMANTIC_BAGGER_PARAMS = {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_leaf": 4,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    # 4. Contextual Branch

    # Metadata Anchor (Logistic Regression)
    # High-bias regularizer
    METADATA_ANCHOR_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
    }

    # Temporal Booster (LightGBM)
    # Captures non-linear temporal drift
    TEMPORAL_BOOSTER_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 15,  # Restricted complexity for metadata
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "objective": "binary",
        "metric": "auc",
        "verbose": -1,
    }

    # =========================================================================
    # Model Hyperparameters (Level 2 Meta Learner)
    # =========================================================================

    # Logistic Regression Stacker
    META_LEARNER_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "random_state": RANDOM_STATE,
    }
