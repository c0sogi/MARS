import os


class Config:
    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_31"
    SUBMISSION_DIR = "./submission"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Global Settings
    # =========================================================================
    RANDOM_SEED = 42
    TARGET_COL = "requester_received_pizza"
    ID_COL = "request_id"

    # Text Columns
    # Using edit_aware to prevent leakage from edits saying "Thanks for pizza"
    TEXT_COL = "request_text_edit_aware"
    TITLE_COL = "request_title"

    # =========================================================================
    # Feature Engineering Configuration
    # =========================================================================

    # Explicit Allow-List for Metadata/Numerical Features
    # Includes Temporal Anchor and User Stats. Excludes retrieval-time leakage.
    NUMERICAL_ALLOW_LIST = [
        "unix_timestamp_of_request_utc",  # Critical Temporal Anchor
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

    # NLP / Text Processing Parameters
    TEXT_PARAMS = {
        "max_features": 3000,
        "ngram_range": (1, 2),
        "min_df": 5,
        "sublinear_tf": True,
        "stop_words": "english",
    }

    # Embedding Model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================

    # 1. Sparse Lexical & Behavioral Branch (Random Forest)
    # Uses mild regularization to capture specific keywords/subreddits
    SPARSE_RF_PARAMS = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbose": 0,
    }

    # 2. Dense Semantic Branch (Random Forest)
    # Uses STRICT regularization to prevent overfitting on continuous embeddings
    DENSE_RF_PARAMS = {
        "n_estimators": 300,
        "max_depth": 12,  # Strict depth constraint
        "min_samples_leaf": 4,  # Higher leaf requirement
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbose": 0,
    }

    # 3. Dense Semantic Branch (XGBoost)
    # High capacity with early stopping
    XGB_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "scale_pos_weight": 3.0,  # Approx ratio of neg/pos (75/25)
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
    }

    # Early stopping rounds for XGBoost
    XGB_EARLY_STOPPING_ROUNDS = 50

    # 4. Contextual Branch & Meta-Learner (Logistic Regression)
    # High bias regularizer / Calibrator
    LINEAR_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
    }
