import os
import numpy as np


class Config:
    """
    Configuration for the Dual-Backbone Consensus Ensemble (DBCE) solution.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    N_JOBS = 12  # Utilizing available 12 vCPUs

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = "./working/idea_26"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data File Paths
    TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

    # Metadata Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache File Names
    CACHE_EMBEDDINGS_A_PREFIX = "embeddings_minilm_"
    CACHE_EMBEDDINGS_B_PREFIX = "embeddings_mpnet_"
    CACHE_FEATURES_PREFIX = "features_"

    # ==========================================
    # Data Configuration
    # ==========================================
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    # Text Columns for Concatenation
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Numerical Metadata Columns
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
    # Model Architecture
    # ==========================================
    # Branch A: Compact Semantics
    MODEL_A_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_A_DIM = 384

    # Branch B: Deep Semantics
    MODEL_B_NAME = "sentence-transformers/all-mpnet-base-v2"
    MODEL_B_RAW_DIM = 768
    MODEL_B_PCA_COMPONENTS = 200  # Projected dimension

    # ==========================================
    # Training Configuration
    # ==========================================
    N_FOLDS = 5
    N_ESTIMATORS_BAGGING = 20

    # Logistic Regression Hyperparameters
    # High-Regularization Regime: 1e-4 to 10.0
    LR_PARAM_GRID = {
        "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "class_weight": ["balanced", None],
        "penalty": ["l2"],
        "solver": ["lbfgs"],  # Efficient for L2
        "max_iter": [2000],
    }
