import os


class Config:
    # --------------------------------------------------------------------------
    # Project Paths
    # --------------------------------------------------------------------------
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working directory for caching processed features
    WORKING_DIR = "./working/idea_2"
    CACHE_DIR = WORKING_DIR

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Global Settings
    # --------------------------------------------------------------------------
    SEED = 42
    N_JOBS = 12  # Utilize available vCPUs

    # --------------------------------------------------------------------------
    # Data Columns
    # --------------------------------------------------------------------------
    TARGET_COL = "requester_received_pizza"
    ID_COL = "request_id"

    # Text Columns for NLP
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Subreddit column (list of strings) to be flattened
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # Numerical Columns (Strictly Request-Time Features to avoid leakage)
    # Excludes all *_at_retrieval columns
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
        "unix_timestamp_of_request_utc",
    ]

    # Derived Features to be generated during processing
    DERIVED_NUMERICAL_COLS = [
        "request_hour",
        "request_day_of_week",
        "text_word_count",
        "title_word_count",
    ]

    # --------------------------------------------------------------------------
    # Feature Engineering Hyperparameters
    # --------------------------------------------------------------------------
    # TF-IDF Vectorization
    TFIDF_MAX_FEATURES = 3000
    TFIDF_NGRAM_RANGE = (1, 2)

    # Semantic Embeddings (Transformer)
    TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384  # Output dimension for all-MiniLM-L6-v2
    EMBEDDING_BATCH_SIZE = 32

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    # Lexical-Linear Branch (Logistic Regression)
    LR_PARAMS = {
        "C": 0.1,  # Strong regularization
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": SEED,
        "max_iter": 1000,
    }

    # Lexical-Tree & Semantic-Tree Branches (Random Forest)
    RF_PARAMS = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": N_JOBS,
        "verbose": 0,
    }
