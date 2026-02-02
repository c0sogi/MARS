import os


class Config:
    """
    Configuration class for the Latent Persona Augmented Dense Fusion (LPADF) strategy.
    Defines global constants, hyperparameters, and file paths.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42

    # ==========================================
    # Directory Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_18"
    SUBMISSION_DIR = "./submission"

    # ==========================================
    # File Paths
    # ==========================================
    # Raw JSON Data
    TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

    # Metadata CSVs
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache Paths
    CACHE_DIR = WORKING_DIR

    # ==========================================
    # Feature Engineering Hyperparameters
    # ==========================================
    # View 1: Semantic Text
    SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # View 2: User Persona (LSA)
    LSA_N_COMPONENTS = 16
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # View 3: Robust Metadata
    # All numerical metadata available at request time, including timestamp
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
        "unix_timestamp_of_request",
    ]

    # ==========================================
    # Model & Training Hyperparameters
    # ==========================================
    N_FOLDS = 5

    # Logistic Regression Grid Search Space
    # Searching High-Regularization Regime (C <= 10.0)
    LR_C_CANDIDATES = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    LR_CLASS_WEIGHTS = ["balanced", None]
    LR_PENALTY = "l2"  # Ridge Regression
    LR_MAX_ITER = 1000

    # Bagging Ensemble Settings
    BAGGING_N_ESTIMATORS = 10
    BAGGING_MAX_SAMPLES = 1.0  # Standard bootstrapping

    # ==========================================
    # Setup
    # ==========================================
    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
