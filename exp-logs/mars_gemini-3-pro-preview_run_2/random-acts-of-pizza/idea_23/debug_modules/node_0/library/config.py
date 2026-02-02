import os


class Config:
    """
    Configuration for the Stratified Random Subspace Linear Ensemble (SRSLE) solution.
    """

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sampleSubmission.csv")

    METADATA_DIR = "./metadata"
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching intermediate files (embeddings, features)
    WORKING_DIR = "./working/idea_23"

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    # Set to a positive integer (e.g., 100) to limit dataset size for debugging; None for full run
    MAX_SAMPLES = None

    # ==========================================
    # Feature Configuration
    # ==========================================
    # Text columns to be concatenated and encoded
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Numerical Metadata columns to retain (100% retention in ensemble)
    # Selected based on high signal strength and importance of "Effort" and "Temporal" features
    NUMERIC_COLS = [
        "unix_timestamp_of_request",
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
    ]

    TARGET_COL = "requester_received_pizza"
    ID_COL = "request_id"

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Backbone for text embeddings (Frozen)
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    # Cross-Validation Strategy
    N_FOLDS = 5

    # Ensemble Strategy (Stratified Random Subspace Bagging)
    N_ESTIMATORS = 50  # Number of base learners per fold
    SUBSPACE_FRACTION = (
        0.5  # Fraction of text embedding dimensions to sample per learner
    )

    # Base Learner (Logistic Regression) Hyperparameter Grid
    # We search the high-regularization regime to prevent overfitting on high-dim text
    LR_C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    LR_CLASS_WEIGHTS = ["balanced", None]

    # Preprocessing
    QUANTILE_OUTPUT_DIST = "normal"  # RankGauss transformation for metadata

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
