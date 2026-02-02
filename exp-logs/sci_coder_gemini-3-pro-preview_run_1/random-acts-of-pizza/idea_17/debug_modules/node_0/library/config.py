import os


class Config:
    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_17"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data File Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache File Paths (for deterministic processing)
    # These files will store processed features to avoid re-computation
    RF_FEATURES_PATH = os.path.join(WORKING_DIR, "rf_features.npz")
    MLP_FEATURES_PATH = os.path.join(WORKING_DIR, "mlp_features.npz")

    # ==========================================
    # Column Definitions
    # ==========================================
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    # Text Input Columns
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Subreddit History Column (List of strings in JSON, string representation in CSV)
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # Numerical Metadata Columns
    # Strictly using features available at the time of request to prevent leakage
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
    # Global Hyperparameters
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to limit dataset size for rapid prototyping
    DEBUG_SAMPLE_SIZE = 100

    # ==========================================
    # Stream A: Random Forest Configuration
    # ==========================================
    # Text Vectorization (Title + Body)
    RF_TFIDF_MAX_FEATURES = 5000
    RF_TFIDF_NGRAM_RANGE = (1, 2)

    # Direct Subreddit Vectorization
    RF_SUBREDDIT_TFIDF_MIN_DF = 2
    RF_SUBREDDIT_TFIDF_BINARY = False

    # Random Forest Model Parameters
    RF_N_ESTIMATORS = 500
    RF_MAX_DEPTH = None
    RF_MIN_SAMPLES_SPLIT = 2
    RF_CLASS_WEIGHT = "balanced"
    RF_N_JOBS = -1

    # ==========================================
    # Stream B: Attention-Gated MLP Configuration
    # ==========================================
    # Architecture
    SBERT_MODEL = "all-MiniLM-L6-v2"
    SBERT_DIM = 384
    MLP_HIDDEN_DIM = 256
    MLP_DROPOUT = 0.3

    # Training Parameters
    MLP_LR = 1e-4
    MLP_WEIGHT_DECAY = 1e-5
    MLP_BATCH_SIZE = 32
    MLP_EPOCHS = 50
    MLP_PATIENCE = 15  # Early stopping patience

    # ==========================================
    # Ensemble Configuration
    # ==========================================
    ENSEMBLE_WEIGHTS = {"rf": 0.5, "mlp": 0.5}
