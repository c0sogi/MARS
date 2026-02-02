import os


class Config:
    # =========================================================================
    # PATHS AND DIRECTORIES
    # =========================================================================
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_29")
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Sample Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sampleSubmission.csv")

    # Output Submission
    SUBMISSION_OUTPUT_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # GLOBAL SETTINGS
    # =========================================================================
    SEED = 42
    N_FOLDS = 5
    DEBUG_SAMPLE_SIZE = (
        None  # Set to an integer (e.g., 100) for debugging, None for full run
    )

    # =========================================================================
    # DATA DEFINITIONS
    # =========================================================================
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    # Text Columns
    TEXT_COL = "request_text_edit_aware"  # Use edit-aware to prevent leakage
    TITLE_COL = "request_title"

    # Behavioral/History Columns
    SUBREDDIT_LIST_COL = "requester_subreddits_at_request"

    # Generated Feature Names
    CROSS_MODAL_SIM_COL = "cross_modal_similarity"

    # Numerical Features Allow-List
    # Strictly excludes retrieval-time features to prevent leakage.
    NUMERICAL_FEATURES = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        "unix_timestamp_of_request",
    ]

    # =========================================================================
    # FEATURE ENGINEERING HYPERPARAMETERS
    # =========================================================================
    # TF-IDF Settings (Sparse Lexical & Behavioral)
    TFIDF_PARAMS = {
        "max_features": 3000,
        "ngram_range": (1, 2),
        "min_df": 5,
        "sublinear_tf": True,
        "stop_words": "english",
    }

    # Dense Embedding Settings
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    BATCH_SIZE = 32

    # =========================================================================
    # MODEL HYPERPARAMETERS
    # =========================================================================

    # Level 1: Random Forest (Lexical Bagger, Community Bagger, Semantic Bagger)
    RF_PARAMS = {
        "n_estimators": 100,
        "min_samples_leaf": 2,  # Regularization to prevent overfitting
        "max_depth": None,
        "random_state": SEED,
        "n_jobs": -1,
        "class_weight": "balanced",
    }

    # Level 1: XGBoost (Semantic Booster)
    XGB_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": SEED,
        "n_jobs": -1,
        "eval_metric": "auc",
        "tree_method": "hist",  # Faster training
        "early_stopping_rounds": 50,
    }

    # Level 1: Logistic Regression (Metadata Anchor)
    LR_ANCHOR_PARAMS = {
        "C": 1.0,
        "solver": "liblinear",
        "penalty": "l2",
        "max_iter": 1000,
        "random_state": SEED,
        "class_weight": "balanced",
    }

    # Level 2: Meta-Learner (Logistic Regression)
    META_LEARNER_PARAMS = {
        "C": 0.5,  # Stronger regularization for meta-learner
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": SEED,
    }

    # =========================================================================
    # CACHE FILE NAMES
    # =========================================================================
    # These keys map to filenames in the CACHE_DIR
    CACHE_FILES = {
        "train_features": "train_features_full.parquet",
        "val_features": "val_features_full.parquet",
        "test_features": "test_features_full.parquet",
        "train_embeddings": "train_embeddings.npy",
        "val_embeddings": "val_embeddings.npy",
        "test_embeddings": "test_embeddings.npy",
        "tfidf_model": "tfidf_vectorizer.joblib",
        "meta_model": "meta_learner.joblib",
    }
