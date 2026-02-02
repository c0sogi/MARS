import os
import numpy as np


class Config:
    # ==========================================
    # Global Settings
    # ==========================================
    RANDOM_SEED = 42
    N_JOBS = 12  # Utilizing available vCPUs

    # ==========================================
    # Paths
    # ==========================================
    # Read-only input directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Caching (Idea 35 specific)
    WORKING_DIR = "./working/idea_35"

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Backbones
    # ==========================================
    # Anchor Backbone (Primary, Low Variance)
    ANCHOR_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    ANCHOR_DIM = 384

    # Auxiliary Backbone (Residual, High Capacity)
    AUX_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    AUX_DIM = 768

    # Dimensionality Reduction for Residuals
    PCA_COMPONENTS = 50

    # ==========================================
    # Feature Definitions
    # ==========================================
    # Text Columns to Concatenate
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Numerical Metadata (Strictly excluding retrieval-time leakage)
    METADATA_COLS = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        "unix_timestamp_of_request",  # Explicitly included per strategy
    ]

    TARGET_COL = "requester_received_pizza"
    ID_COL = "request_id"

    # ==========================================
    # Training Configuration
    # ==========================================
    N_SPLITS = 5  # Stratified K-Fold

    # Bagging Ensemble Settings
    N_ESTIMATORS = 20  # Number of base estimators in BaggingClassifier

    # Hyperparameter Search Space for Logistic Regression (Ridge)
    # High-Regularization Regime to Standard
    PARAM_GRID = {
        "C": np.logspace(-4, 1, 6).tolist(),  # 0.0001 to 10.0
        "class_weight": ["balanced", None],
        # Penalty is fixed to 'l2' in implementation
    }

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
