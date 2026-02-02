import os
import numpy as np


class Config:
    """
    Configuration for Latent Persona Augmented Dense Fusion (LPADF) strategy.
    Defines paths, hyperparameters, and feature engineering settings.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_19"
    SUBMISSION_DIR = "./submission"

    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Splits
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Global Settings
    # ==========================================
    RANDOM_SEED = 42
    N_JOBS = 12  # Utilize available vCPUs

    # ==========================================
    # Feature Engineering Parameters
    # ==========================================
    # View 1: Semantic Request Content (SBERT)
    SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # View 2: Latent User Persona (LSA)
    SUBREDDIT_COL = "requester_subreddits_at_request"
    LSA_N_COMPONENTS = 16  # Asymmetric Dimensionality Reduction target

    # View 3: Robust Metadata
    # Explicitly including unix_timestamp_of_request per strategy
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
    # Model Hyperparameters
    # ==========================================
    # Stratified Cross-Validation
    N_FOLDS = 5

    # Ensemble: Bagged Logistic Regression
    N_ESTIMATORS = 20  # Number of bagging estimators

    # Grid Search Space for Logistic Regression (Ridge)
    # Searching High-Regularization Regime: 1e-4 to 10.0
    LR_C_RANGE = np.logspace(-4, 1, 6).tolist()  # [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    LR_CLASS_WEIGHTS = ["balanced", None]

    # ==========================================
    # Execution Control
    # ==========================================
    LOAD_CACHED_DATA = True  # Enable caching mechanism
    DEBUG = False  # Set to True for fast debugging runs
    DEBUG_SIZE = 100  # Sample size when DEBUG is True
