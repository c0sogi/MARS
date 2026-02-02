import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_56"
    OUTPUT_DIR = "./submission"

    # Input Files (Metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

    # Cache Paths (Deterministic Processing)
    # Using .parquet for tabular data and .npz for numpy arrays/tensors
    CACHE_RF_TRAIN = os.path.join(WORKING_DIR, "rf_train.parquet")
    CACHE_RF_VAL = os.path.join(WORKING_DIR, "rf_val.parquet")
    CACHE_RF_TEST = os.path.join(WORKING_DIR, "rf_test.parquet")

    CACHE_MLP_TRAIN = os.path.join(WORKING_DIR, "mlp_train.npz")
    CACHE_MLP_VAL = os.path.join(WORKING_DIR, "mlp_val.npz")
    CACHE_MLP_TEST = os.path.join(WORKING_DIR, "mlp_test.npz")

    # --------------------------------------------------------------------------
    # Global Settings
    # --------------------------------------------------------------------------
    RANDOM_SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Set to an integer (e.g., 100) for debugging, or None for full dataset
    MAX_SAMPLES = None

    # --------------------------------------------------------------------------
    # Feature Engineering Configuration
    # --------------------------------------------------------------------------
    # Text Processing
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
    TFIDF_VOCAB_SIZE = 5000

    # Top-K Subreddits for Binary Indicators
    TOP_K_SUBREDDITS = 50

    # Columns
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Numerical columns safe from leakage (at_request only)
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

    # List columns (for history processing)
    LIST_COLS = ["requester_subreddits_at_request"]

    # Interaction Features
    # Metrics to cross-product with Consistency Scalars in the RF stream
    INTERACTION_CREDIBILITY_METRICS = [
        "requester_account_age_in_days_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_number_of_posts_on_raop_at_request",
    ]

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------

    # Stream A: Interaction-Enhanced Random Forest
    RF_ESTIMATORS = 500
    RF_MAX_DEPTH = None  # Allow full growth
    RF_MIN_SAMPLES_LEAF = 1  # Low regularization to capture fine signals
    RF_CLASS_WEIGHT = "balanced"
    RF_N_JOBS = -1

    # Stream B: Non-Linear Orthogonal Skip-Gated MLP
    MLP_HIDDEN_DIM = 128
    MLP_DROPOUT_EMB = 0.5
    MLP_DROPOUT_DENSE = 0.2
    MLP_LEARNING_RATE = 1e-4
    MLP_WEIGHT_DECAY = 1e-4
    MLP_BATCH_SIZE = 32
    MLP_EPOCHS = 50
    MLP_PATIENCE = 15

    # Ensemble Weights
    WEIGHT_RF = 0.5
    WEIGHT_MLP = 0.5

    @classmethod
    def setup(cls):
        """Ensure working and output directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
