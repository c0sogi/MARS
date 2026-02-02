import os


class Config:
    """
    Configuration for the Hybrid Ensemble with Predictive-Relevance Profiling
    and Dropout-Stabilized Dual-Attention.
    """

    # ==========================================
    # PATHS & DIRECTORIES
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and models
    # Using a specific subdirectory for this idea to avoid conflicts
    WORKING_DIR = "./working/idea_34"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data File Paths (using metadata splits)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # DATA CONFIGURATION
    # ==========================================
    RANDOM_STATE = 42
    TARGET_COL = "requester_received_pizza"

    # Columns to exclude from features:
    # 1. IDs and Artifacts (request_id, source_file)
    # 2. Leakage (giver_username, requester_user_flair)
    # 3. Redundant/Raw Text (request_text - we use edit_aware)
    # Note: 'requester_user_flair' is leakage because 'shroom' = received pizza.
    LEAKAGE_COLUMNS = [
        "giver_username_if_known",
        "request_id",
        "source_file",
        "requester_user_flair",
        "request_text",  # Using request_text_edit_aware instead
        "unix_timestamp_of_request",  # Redundant with UTC version
    ]

    # Text Columns
    TEXT_COL_TITLE = "request_title"
    TEXT_COL_BODY = "request_text_edit_aware"

    # Subreddit History Column (contains list of strings)
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # ==========================================
    # FEATURE ENGINEERING HYPERPARAMETERS
    # ==========================================
    # Sentence-BERT model for semantic embeddings
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

    # TF-IDF Vectorization
    TFIDF_VOCAB_SIZE = 5000

    # Predictive Profiling
    # Number of top subreddits to select based on Mutual Information with target
    TOP_K_MI_SUBREDDITS = 50

    # ==========================================
    # MODEL HYPERPARAMETERS
    # ==========================================
    # Stream A: Predictive-Relevance Random Forest
    RF_N_ESTIMATORS = 500
    RF_CLASS_WEIGHT = "balanced"
    RF_MIN_SAMPLES_LEAF = 1
    RF_MAX_DEPTH = None
    RF_N_JOBS = -1

    # Stream B: Dropout-Stabilized Dual-Query MLP
    MLP_HIDDEN_DIM = 256
    MLP_PROJECTION_DIM = 128

    # Regularization (Dropout-Only Regime)
    MLP_DROPOUT_EMBEDDING = 0.5  # Higher dropout for embeddings
    MLP_DROPOUT_DENSE = 0.2  # Standard dropout for dense layers
    MLP_USE_BATCH_NORM = False  # Strictly False as per "Dropout-Stabilized" idea

    # Optimization
    MLP_LEARNING_RATE = 1e-4
    MLP_WEIGHT_DECAY = 1e-5
    MLP_BATCH_SIZE = 32
    MLP_EPOCHS = 50
    MLP_PATIENCE = 15  # High patience for stability

    # ==========================================
    # ENSEMBLE CONFIGURATION
    # ==========================================
    # Simple Weighted Average
    ENSEMBLE_WEIGHT_RF = 0.5
    ENSEMBLE_WEIGHT_MLP = 0.5
