import os


class Config:
    """
    Configuration class for the Pizza Request Success Prediction project.
    Implements settings for the Differentially-Regularized Bagged Linear Ensemble.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    RANDOM_SEED = 42

    # ==========================================
    # File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"
    SUBMISSION_DIR = "./submission"

    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata (Stratified Splits)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Outputs
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    CACHE_DIR = WORKING_DIR  # Directory for caching processed features (parquet/npy)

    # ==========================================
    # Data Processing Configuration
    # ==========================================
    # Text columns to be concatenated and embedded
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Target label column
    TARGET_COL = "requester_received_pizza"

    # ID column
    ID_COL = "request_id"

    # Columns to strictly exclude from numeric feature set
    # (IDs, Text, Target, Username, etc.)
    EXCLUDE_COLS = [
        "request_id",
        "requester_received_pizza",
        "request_text",
        "request_text_edit_aware",
        "request_title",
        "giver_username_if_known",
        "requester_username",
        "requester_user_flair",
        "requester_subreddits_at_request",
        "post_was_edited",  # Boolean, often handled separately or excluded if low signal
        # Exclude retrieval-time features (leakage/future data not in test set)
        "number_of_downvotes_of_request_at_retrieval",
        "number_of_upvotes_of_request_at_retrieval",
        "request_number_of_comments_at_retrieval",
        "requester_account_age_in_days_at_retrieval",
        "requester_days_since_first_post_on_raop_at_retrieval",
        "requester_number_of_comments_at_retrieval",
        "requester_number_of_comments_in_raop_at_retrieval",
        "requester_number_of_posts_at_retrieval",
        "requester_number_of_posts_on_raop_at_retrieval",
        "requester_upvotes_minus_downvotes_at_retrieval",
        "requester_upvotes_plus_downvotes_at_retrieval",
    ]

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # 1. Text Embedding (Frozen Backbone)
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    EMBEDDING_BATCH_SIZE = 32

    # 2. Classifier: Bagged Logistic Regression with Differential Regularization
    # Cross-Validation Strategy
    N_FOLDS = 5

    # Grid Search Space for Hyperparameter Optimization
    # 'alpha': Scaling factor for metadata features (Differential Regularization)
    # 'C': Inverse regularization strength for Logistic Regression
    GRID_SEARCH_PARAMS = {
        "C": [1e-4, 1e-3, 0.01, 0.1, 1.0],
        "alpha": [1.0, 2.0, 5.0, 10.0],
        "class_weight": ["balanced", None],
    }

    # Bagging Ensemble Settings
    BAGGING_N_ESTIMATORS = 10  # Number of base estimators in the ensemble
    BAGGING_MAX_SAMPLES = 0.8  # Fraction of samples to draw for each base estimator

    # Base Estimator (Logistic Regression) Settings
    LOGREG_MAX_ITER = 1000
    LOGREG_SOLVER = "liblinear"  # Efficient for high-dimensional data
    LOGREG_PENALTY = "l2"

    # ==========================================
    # Execution & Debugging
    # ==========================================
    # Set to True to run on a small subset of data for pipeline verification
    DEBUG = False

    # Number of samples to use when DEBUG is True
    DEBUG_SAMPLE_SIZE = 100

    @classmethod
    def ensure_directories(cls):
        """Creates necessary working and submission directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
