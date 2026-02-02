import os


class Config:
    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_27"
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data File Paths
    # -------------------------------------------------------------------------
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Column Definitions
    # -------------------------------------------------------------------------
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    # Text Inputs
    # Use edit-aware text to prevent leakage from "EDIT: Thanks for pizza"
    TEXT_COL = "request_text_edit_aware"
    TITLE_COL = "request_title"

    # Behavioral/History Input
    HISTORY_COL = "requester_subreddits_at_request"

    # Allow-listed Numerical Features (Metadata)
    # Strictly excluding:
    # 1. Retrieval-time features (Leakage)
    # 2. Derived text length features (Noise/Overfitting)
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
        "unix_timestamp_of_request",  # Temporal signal
    ]

    # -------------------------------------------------------------------------
    # Model & Processing Configuration
    # -------------------------------------------------------------------------
    SEED = 42

    # Embedding Model
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    # TF-IDF Configuration (Sparse Features)
    TFIDF_PARAMS = {
        "max_features": 3000,
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "min_df": 5,
        "stop_words": "english",
    }

    # Random Forest Configuration (Base Learners: Sparse & Dense Bagging)
    # Key: class_weight='balanced' and min_samples_leaf=2 for regularization
    RF_PARAMS = {
        "n_estimators": 100,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": 0,
    }

    # XGBoost Configuration (Base Learner: Dense Boosting)
    # Key: scale_pos_weight=3.0 for ~1:3 imbalance, high estimators for early stopping
    XGB_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 3.0,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": 0,
        "early_stopping_rounds": 50,
    }

    # Logistic Regression Configuration (Base Learner: Contextual & Meta Learner)
    LR_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": SEED,
        "max_iter": 1000,
    }

    # -------------------------------------------------------------------------
    # Caching Paths (Artifacts)
    # -------------------------------------------------------------------------
    # Paths for caching processed feature matrices to disk
    CACHE_TRAIN_META = os.path.join(WORKING_DIR, "X_train_meta.npy")
    CACHE_VAL_META = os.path.join(WORKING_DIR, "X_val_meta.npy")
    CACHE_TEST_META = os.path.join(WORKING_DIR, "X_test_meta.npy")

    CACHE_TRAIN_TEXT_TFIDF = os.path.join(WORKING_DIR, "X_train_text_tfidf.npz")
    CACHE_VAL_TEXT_TFIDF = os.path.join(WORKING_DIR, "X_val_text_tfidf.npz")
    CACHE_TEST_TEXT_TFIDF = os.path.join(WORKING_DIR, "X_test_text_tfidf.npz")

    CACHE_TRAIN_HIST_TFIDF = os.path.join(WORKING_DIR, "X_train_hist_tfidf.npz")
    CACHE_VAL_HIST_TFIDF = os.path.join(WORKING_DIR, "X_val_hist_tfidf.npz")
    CACHE_TEST_HIST_TFIDF = os.path.join(WORKING_DIR, "X_test_hist_tfidf.npz")

    CACHE_TRAIN_EMBED = os.path.join(WORKING_DIR, "X_train_embed.npy")
    CACHE_VAL_EMBED = os.path.join(WORKING_DIR, "X_val_embed.npy")
    CACHE_TEST_EMBED = os.path.join(WORKING_DIR, "X_test_embed.npy")

    CACHE_Y_TRAIN = os.path.join(WORKING_DIR, "y_train.npy")
    CACHE_Y_VAL = os.path.join(WORKING_DIR, "y_val.npy")
