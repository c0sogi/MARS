import os


class Config:
    # ==========================================
    # Global Settings
    # ==========================================
    RANDOM_SEED = 42
    N_FOLDS = 5
    DEBUG = False  # Set to True for debugging with smaller dataset
    DEV_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_25"
    SUBMISSION_DIR = "./submission"

    # Create directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Submission File
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Caching Paths
    # ==========================================
    # Embeddings
    TRAIN_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "train_embeddings.npy")
    VAL_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "val_embeddings.npy")
    TEST_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "test_embeddings.npy")

    # Processed Features (Parquet)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Text Encoder
    SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    SBERT_BATCH_SIZE = 32

    # Homophily / Target Encoding
    # Inner folds for OOF target encoding generation within the training set
    INNER_CV_FOLDS = 5

    # Classifier (Logistic Regression)
    LR_C = 1.0  # Inverse of regularization strength (smaller = stronger regularization)
    LR_MAX_ITER = 1000
    LR_SOLVER = "lbfgs"
    LR_CLASS_WEIGHT = "balanced"  # Can be None or 'balanced'

    # Bagging Ensemble
    BAGGING_N_ESTIMATORS = 10
    BAGGING_MAX_SAMPLES = 0.8  # Fraction of samples to draw for each base estimator
    BAGGING_MAX_FEATURES = 1.0  # Fraction of features to draw

    # ==========================================
    # Feature Definitions
    # ==========================================
    TARGET_COL = "requester_received_pizza"

    # Text Features (View 1)
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Homophily Features (View 2)
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # Numeric Metadata Features (View 3)
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
