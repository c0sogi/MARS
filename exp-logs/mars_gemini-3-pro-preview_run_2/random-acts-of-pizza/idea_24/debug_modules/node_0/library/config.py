import os


class Config:
    """
    Configuration for the Passthrough-Stacked Hybrid Linear Ensemble (PSHLE).
    Centralizes paths, feature definitions, and model hyperparameters.
    """

    # --- General Configuration ---
    RANDOM_SEED = 42
    N_JOBS = 12  # Available vCPUs
    N_FOLDS = 5  # CV Strategy

    # Debugging: Set to a small integer (e.g., 100) to run on a subset, or None for full data
    DEBUG_SAMPLE_SIZE = None

    # --- Path Definitions ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_24"
    SUBMISSION_DIR = "./submission"

    # Raw Data
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata (Splits)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (Parquet/NPY for fast I/O)
    CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features.parquet")
    CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "val_features.parquet")
    CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "test_features.parquet")

    # --- Column Definitions ---
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    # Text Features (Concatenated Title + Body)
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # History Feature (List of Subreddits for Bayesian Encoding)
    HISTORY_COL = "requester_subreddits_at_request"

    # Numerical Metadata (Passthrough Features)
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
        "unix_timestamp_of_request",  # Explicitly included per strategy
    ]

    # --- Model Configuration ---

    # Text Encoding
    SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # History Expert: Bayesian Target Encoding
    HISTORY_SMOOTHING = 10.0
    HISTORY_MIN_SAMPLES = 1

    # Stage 1: Text Expert (Bagged Logistic Regression)
    # Strategy: High Regularization (Small C) to compress high-dim text to scalar probability
    TEXT_EXPERT_GRID = {
        "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "class_weight": ["balanced", None],
        "solver": ["liblinear"],  # Efficient for high-dimensional sparse/dense
        "penalty": ["l2"],
        "random_state": [RANDOM_SEED],
    }

    # Stage 2: Meta-Learner (Bagged Logistic Regression)
    # Strategy: Passthrough Stacking. Input is [Text_Prob, History_Score, Metadata_RankGauss]
    # Lower Regularization (Higher C) allowed as inputs are low-dim and high-signal
    META_LEARNER_GRID = {
        "C": [0.1, 1.0, 10.0, 100.0],
        "class_weight": ["balanced", None],
        "solver": ["lbfgs"],
        "max_iter": [1000],
        "random_state": [RANDOM_SEED],
    }

    @classmethod
    def setup(cls):
        """
        Creates necessary working directories.
        Should be called at the start of the pipeline.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
