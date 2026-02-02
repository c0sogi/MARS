import os


class Config:
    """
    Configuration for the Context-Aware Asymmetric Early Fusion (CAAEF) strategy.
    Centralizes paths, hyperparameters, and feature definitions.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a subset of data
    MAX_SAMPLES = 100 if DEBUG else None  # Limit samples for debugging

    # ==========================================
    # Directory Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_36"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # File Paths
    # ==========================================
    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sampleSubmission.csv")

    # Metadata
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Output Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cached Artifacts (Embeddings & Features)
    # We use .npy for embeddings and .parquet for tabular features
    TRAIN_EMB_ANCHOR_PATH = os.path.join(WORKING_DIR, "train_emb_anchor.npy")
    TRAIN_EMB_AUX_PATH = os.path.join(WORKING_DIR, "train_emb_aux.npy")

    VAL_EMB_ANCHOR_PATH = os.path.join(WORKING_DIR, "val_emb_anchor.npy")
    VAL_EMB_AUX_PATH = os.path.join(WORKING_DIR, "val_emb_aux.npy")

    TEST_EMB_ANCHOR_PATH = os.path.join(WORKING_DIR, "test_emb_anchor.npy")
    TEST_EMB_AUX_PATH = os.path.join(WORKING_DIR, "test_emb_aux.npy")

    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Backbone Models
    ANCHOR_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384d
    AUX_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 768d

    # Feature Engineering
    AUX_PCA_COMPONENTS = 50  # Compress Aux backbone to 50d
    INTERACTION_TOP_K = 5  # Use top 5 PCA components for interactions
    INTERACTION_DEGREE = 2  # Degree for PolynomialFeatures

    # Training
    N_FOLDS = 5
    N_BAGGING_ESTIMATORS = 20  # Number of estimators in BaggingClassifier

    # Hyperparameter Grid for Logistic Regression (Base Estimator)
    # Note: Keys are prefixed with 'base_estimator__' for use in BaggingClassifier or Pipeline
    PARAM_GRID = {
        "base_estimator__C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "base_estimator__class_weight": ["balanced", None],
    }

    # ==========================================
    # Feature Definitions
    # ==========================================
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    # Text Columns for Embedding
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Numerical Metadata Columns (View 3)
    # Explicitly selected based on data analysis and strategy
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
