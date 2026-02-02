import os
import torch


class Config:
    """
    Configuration for the Hybrid Ensemble pipeline (Idea 32).
    Includes settings for Paths, Feature Engineering, Random Forest, and MLP models.
    """

    # =========================================================================
    # System & Reproducibility
    # =========================================================================
    PROJECT_NAME = "idea_32"
    RANDOM_SEED = 42

    # Debugging: Set to True to run on a small subset of data for testing
    DEBUG = False
    DEBUG_SIZE = 100

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)
    SUBMISSION_DIR = "./submission"

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Data (Metadata CSVs)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (for deterministic data processing)
    # We use Parquet for dataframes and .pth/.joblib for models
    CACHE_TRAIN_PROCESSED = os.path.join(WORKING_DIR, "train_processed.parquet")
    CACHE_VAL_PROCESSED = os.path.join(WORKING_DIR, "val_processed.parquet")
    CACHE_TEST_PROCESSED = os.path.join(WORKING_DIR, "test_processed.parquet")

    # Model Artifacts
    MODEL_RF_PATH = os.path.join(WORKING_DIR, "rf_model.joblib")
    MODEL_MLP_PATH = os.path.join(WORKING_DIR, "mlp_best_model.pth")

    # =========================================================================
    # Feature Engineering Hyperparameters
    # =========================================================================
    # Text Embeddings (SBERT)
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
    SBERT_EMBEDDING_DIM = 384

    # TF-IDF (for Random Forest)
    TFIDF_MAX_FEATURES = 5000
    TFIDF_NGRAM_RANGE = (1, 2)

    # User History & Community Profiling
    TOP_K_SUBREDDITS = 50  # Number of top subreddits for binary indicators
    MAX_HISTORY_LEN = 50  # Max sequence length of user history for MLP

    # Metadata
    # Numerical columns to be scaled and used in both models
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
    ]

    # =========================================================================
    # Model A: Peak-Relevance Augmented Random Forest
    # =========================================================================
    RF_N_ESTIMATORS = 500
    RF_MIN_SAMPLES_LEAF = 1
    RF_CLASS_WEIGHT = "balanced"
    RF_N_JOBS = -1  # Use all available cores

    # =========================================================================
    # Model B: Dual-Query Alignment-Gated MLP (Dropout-Only)
    # =========================================================================
    MLP_HIDDEN_DIM = 256
    MLP_DROPOUT_EMB = 0.5  # High dropout on embeddings
    MLP_DROPOUT_DENSE = 0.2  # Moderate dropout on dense layers
    MLP_USE_BATCHNORM = False  # Strictly False to avoid instability (Lesson 00062)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # Weight decay for AdamW
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 15

    # Ensemble Weights (Simple Average)
    ENSEMBLE_WEIGHT_RF = 0.5
    ENSEMBLE_WEIGHT_MLP = 0.5
