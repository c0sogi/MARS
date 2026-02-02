import os


class Config:
    # ==========================================
    # 1. File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate files (parquet/npy)
    WORKING_DIR = "./working/idea_28"

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Input CSV paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Global Settings
    # ==========================================
    RANDOM_STATE = 42

    # Target Column
    TARGET_COL = "requester_received_pizza"

    # ID Column
    ID_COL = "request_id"

    # ==========================================
    # 3. Feature Definitions
    # ==========================================
    # Text Columns used for TF-IDF (RF) and SBERT (MLP)
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Raw Numerical Metadata Columns (Available at Request Time)
    # Note: 'at_retrieval' columns are excluded to prevent data leakage.
    NUMERIC_COLS = [
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

    # New Sentiment Features to be Engineered
    SENTIMENT_COLS = [
        "title_polarity",
        "title_subjectivity",
        "body_polarity",
        "body_subjectivity",
    ]

    # Subreddit History Column (List of strings)
    HISTORY_COL = "requester_subreddits_at_request"

    # ==========================================
    # 4. Model Hyperparameters
    # ==========================================

    # Stream A: Random Forest Hyperparameters
    # High estimators, minimal regularization, balanced weights
    RF_PARAMS = {
        "n_estimators": 500,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": 0,
    }

    # Stream B: MLP Hyperparameters
    # Dual-Query Masked-Attention + Gated Fusion
    MLP_PARAMS = {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 50,
        "patience": 15,  # High patience for stability
        "dropout": 0.3,  # Regularization for embeddings
        "hidden_dim": 128,  # Dimension for internal projections
        "sbert_model": "all-MiniLM-L6-v2",  # Efficient SBERT model
        "max_history_len": 50,  # Max number of subreddits to consider in history sequence
    }

    # TF-IDF Settings for RF
    TFIDF_PARAMS = {
        "max_features": 5000,
        "stop_words": "english",
        "ngram_range": (1, 2),
        "sublinear_tf": True,
    }

    # Ensemble Weights
    # Simple Weighted Average
    ENSEMBLE_WEIGHTS = {"rf": 0.5, "mlp": 0.5}

    @classmethod
    def setup(cls):
        """Ensure necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup on import
Config.setup()
