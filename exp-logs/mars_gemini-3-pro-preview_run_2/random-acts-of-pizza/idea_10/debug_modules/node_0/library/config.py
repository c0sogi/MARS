import os


class Config:
    """
    Configuration class for the Random Acts of Pizza success prediction task.
    Defines file paths, model hyperparameters, and feature engineering constants.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLES = 100  # Number of samples to use when DEBUG is True

    # ==========================================
    # Directory Structure
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # File Paths
    # ==========================================
    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata (Splits)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Cached Features (Parquet)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Output Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Feature Engineering Constants
    # ==========================================
    # Semantic View: Sentence Transformer Model
    SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Lexical View: TF-IDF for Request Text
    TFIDF_TEXT_PARAMS = {
        "max_features": 2000,
        "min_df": 5,
        "ngram_range": (1, 2),
        "stop_words": "english",
        "norm": "l2",
        "sublinear_tf": True,
    }

    # Community View: TF-IDF for User Subreddits
    TFIDF_SUBREDDIT_PARAMS = {
        "max_features": 500,
        "min_df": 2,
        "ngram_range": (1, 1),
        "analyzer": "word",
        "token_pattern": r"(?u)\b\w+\b",  # Simple alphanumeric tokenization
        "norm": "l2",
    }

    # Columns Definitions
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    # Text columns to be concatenated (Title + Body)
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Column containing list of subreddits
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # Numerical columns for RankGauss (QuantileTransformer)
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

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Logistic Regression Regularization (Inverse Strength)
    # Includes strong regularization (small C) to handle high dimensionality
    LOGREG_C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

    # Bagging Ensemble Configuration
    BAGGING_PARAMS = {
        "n_estimators": 20,
        "max_samples": 0.8,
        "random_state": SEED,
        "n_jobs": -1,
    }
