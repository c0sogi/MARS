import os


class Config:
    """
    Configuration for Symmetric Multi-Modal Stacking Ensemble.
    Defines paths, feature engineering settings, and model hyperparameters.
    """

    # ==========================================
    # Paths & Directories
    # ==========================================
    # Input Metadata (Pre-stratified splits)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working Directories
    WORKING_DIR = "./working/idea_18"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_DIR = os.path.join(WORKING_DIR, "models")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure output directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    N_JOBS = 12  # 12 vCPUs available
    DEVICE = "cuda"  # NVIDIA A100 available

    # ==========================================
    # Data Definitions
    # ==========================================
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    # Text Columns: Use edit-aware text to prevent leakage from "EDIT: Thanks!"
    TEXT_COL = "request_text_edit_aware"
    TITLE_COL = "request_title"

    # Behavioral Columns
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # Leakage Prevention: Exclude retrieval-time stats
    EXCLUDED_SUFFIX = "_at_retrieval"
    EXCLUDED_COLS = [
        "post_was_edited",
        "giver_username_if_known",
        "request_text",  # Use edit_aware version
        "source_file",
        "requester_username",
        "requester_user_flair",  # Flair often indicates success (e.g. 'shroom')
        "unix_timestamp_of_request_utc",  # Redundant
    ]

    # Unified Metadata Vector: Allow-list of dense features
    NUMERICAL_COLS = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        "unix_timestamp_of_request",  # Used for temporal feature extraction
    ]

    # ==========================================
    # Feature Engineering Hyperparameters
    # ==========================================
    # Text - Sparse View (TF-IDF)
    TEXT_TFIDF_PARAMS = {
        "max_features": 3000,
        "ngram_range": (1, 2),
        "stop_words": "english",
        "sublinear_tf": True,
        "min_df": 5,
    }

    # Text - Dense View (Transformer Embeddings)
    MPNET_MODEL_NAME = "all-mpnet-base-v2"
    MAX_SEQ_LENGTH = 512
    BATCH_SIZE = 32

    # Behavior - Sparse View (Subreddit Bag-of-Concepts)
    SUBREDDIT_TFIDF_PARAMS = {
        "max_features": 1000,
        "ngram_range": (1, 1),
        "stop_words": "english",
        "binary": True,  # Membership is binary
    }

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    N_FOLDS = 5
    EARLY_STOPPING_ROUNDS = 50

    # --- Level 1: Base Learners ---

    # 1. Text Modality Branch
    # Lexical Bagger (Random Forest on TF-IDF + Meta)
    RF_LEXICAL_PARAMS = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": N_JOBS,
        "random_state": SEED,
    }

    # Semantic Ensemble (XGBoost on Embeddings + Meta)
    XGB_SEMANTIC_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "max_depth": 4,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "n_jobs": N_JOBS,
        "random_state": SEED,
        "tree_method": "hist",
        "device": DEVICE,
    }

    # Semantic Ensemble (Random Forest on Embeddings + Meta)
    RF_SEMANTIC_PARAMS = {
        "n_estimators": 300,
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "n_jobs": N_JOBS,
        "random_state": SEED,
    }

    # 2. Behavioral Modality Branch
    # Community Bagger (Random Forest on Subreddit TF-IDF + Meta)
    RF_BEHAVIORAL_PARAMS = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": N_JOBS,
        "random_state": SEED,
    }

    # Persona Booster (XGBoost on Subreddit Embeddings + Meta)
    XGB_BEHAVIORAL_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "max_depth": 4,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "n_jobs": N_JOBS,
        "random_state": SEED,
        "tree_method": "hist",
        "device": DEVICE,
    }

    # 3. Contextual Modality Branch
    # Metadata Anchor (Logistic Regression on Meta only)
    LR_CONTEXTUAL_PARAMS = {
        "penalty": "l2",
        "C": 0.1,
        "class_weight": "balanced",
        "solver": "liblinear",
        "random_state": SEED,
        "max_iter": 1000,
    }

    # --- Level 2: Meta-Learner ---
    META_LEARNER_PARAMS = {
        "penalty": "l2",
        "C": 1.0,
        "class_weight": None,  # Calibration is key here
        "solver": "lbfgs",
        "random_state": SEED,
    }
