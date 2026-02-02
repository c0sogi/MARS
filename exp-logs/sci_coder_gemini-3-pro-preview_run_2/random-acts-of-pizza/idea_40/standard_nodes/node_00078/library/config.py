import os
import numpy as np


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_40"
    SUBMISSION_DIR = "./submission"

    # Input Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Submission File
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    N_JOBS = 12  # Utilizing available vCPUs
    DEVICE = "cuda"  # Utilizing NVIDIA A100 GPU

    # ==========================================
    # Models & Features
    # ==========================================
    # Pre-trained Sentence Transformer Models
    # MiniLM for high-resolution field-specific views (Title, Body)
    MODEL_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
    # MPNet for low-resolution global context view
    MODEL_MPNET = "sentence-transformers/all-mpnet-base-v2"

    # Dimensionality Reduction
    PCA_COMPONENTS = 50

    # Feature Columns
    TEXT_COL_TITLE = "request_title"
    TEXT_COL_BODY = "request_text_edit_aware"

    # Numeric metadata features identified during data analysis
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
    # Caching Filenames
    # ==========================================
    # Standardized filenames for cached embeddings to support pre-computation
    CACHE_TRAIN_TITLE_MINILM = os.path.join(WORKING_DIR, "train_title_minilm.npy")
    CACHE_TRAIN_BODY_MINILM = os.path.join(WORKING_DIR, "train_body_minilm.npy")
    CACHE_TRAIN_GLOBAL_MPNET = os.path.join(WORKING_DIR, "train_global_mpnet.npy")

    CACHE_TEST_TITLE_MINILM = os.path.join(WORKING_DIR, "test_title_minilm.npy")
    CACHE_TEST_BODY_MINILM = os.path.join(WORKING_DIR, "test_body_minilm.npy")
    CACHE_TEST_GLOBAL_MPNET = os.path.join(WORKING_DIR, "test_global_mpnet.npy")

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    N_FOLDS = 5
    BAGGING_N_ESTIMATORS = 20

    # Grid Search Parameters for the Bagged Logistic Regression Ensemble
    # Note: scikit-learn >= 1.2 uses 'estimator' instead of 'base_estimator'
    PARAM_GRID = {
        "estimator__C": np.logspace(-4, 1, 10).tolist(),  # Regularization strength
        "estimator__class_weight": ["balanced", None],  # Handle class imbalance
        "estimator__solver": ["lbfgs"],  # Standard solver
        "estimator__max_iter": [1000],  # Ensure convergence
    }
