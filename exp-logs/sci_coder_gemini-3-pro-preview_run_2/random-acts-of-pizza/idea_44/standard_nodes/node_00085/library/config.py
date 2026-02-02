import os


class Config:
    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_44"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_DATA_PATH = os.path.join(INPUT_DIR, "train.json")
    TEST_DATA_PATH = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Global Experiment Settings
    # ==========================================
    SEED = 42
    N_FOLDS = 5
    # Set to an integer (e.g., 100) to run on a small subset for debugging, or None for full run
    DEBUG_SAMPLE_SIZE = None

    # ==========================================
    # 3. Feature Extraction Settings
    # ==========================================
    # Pretrained Models
    ANCHOR_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    AUX_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

    # Asymmetric Dimensionality Reduction Targets
    AUX_TITLE_PCA_COMPONENTS = 20
    AUX_BODY_PCA_COMPONENTS = 30

    # Text Columns
    TEXT_COL_TITLE = "request_title"
    TEXT_COL_BODY = "request_text_edit_aware"

    # Numerical Features (filtered to avoid leakage from retrieval-time data)
    NUMERICAL_FEATURES = [
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
    # 4. Model Hyperparameters
    # ==========================================
    # Bagging Ensemble Settings
    BAGGING_N_ESTIMATORS = 20

    # Grid Search Space for the Base Estimator (Logistic Regression)
    # These parameters are tuned within each fold
    PARAM_GRID = {
        "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "class_weight": ["balanced", None],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
        "max_iter": [2000],  # Increased to ensure convergence
    }
