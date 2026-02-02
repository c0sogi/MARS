import os
import torch


class Config:
    """
    Configuration class for the Pizza Request Success Prediction task.
    Centralizes all file paths, hyperparameters, and model settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Input Data (Metadata CSVs)
    INPUT_DIR = "./metadata"
    TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
    VAL_PATH = os.path.join(INPUT_DIR, "val.csv")
    TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

    # Working & Cache Directories
    # Used for storing processed features (parquet/npy) to avoid re-computation
    WORKING_DIR = "./working/idea_11"
    CACHE_DIR = WORKING_DIR

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Definitions
    # =========================================================================
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"
    TEXT_COL = "request_text_edit_aware"
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # Columns to exclude from numerical features (IDs, Text, Leakage, Target)
    DROP_COLS = [
        "request_id",
        "requester_received_pizza",
        "giver_username_if_known",
        "request_text",
        "request_text_edit_aware",
        "request_title",
        "post_was_edited",
        "source_file",
        "unix_timestamp_of_request",  # Redundant/Leakage
        "unix_timestamp_of_request_utc",
        "requester_username",
        "requester_user_flair",  # Leakage
        "requester_subreddits_at_request",
    ]

    # =========================================================================
    # Feature Engineering Hyperparameters
    # =========================================================================
    # Text Embeddings (SBERT)
    SBERT_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    MAX_TEXT_LEN = 512

    # Stream A: Topic Modeling (K-Means on Subreddit Embeddings)
    NUM_TOPICS = 20  # Number of semantic clusters (K)

    # Stream A: TF-IDF Settings
    TFIDF_MAX_FEATURES = 5000
    TFIDF_NGRAM_RANGE = (1, 2)

    # Stream B: History Attention Settings
    MAX_HISTORY_LEN = 50  # Max number of past subreddits to consider

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================

    # Stream A: Random Forest (The Balanced Partner)
    RF_PARAMS = {
        "n_estimators": 500,
        "class_weight": "balanced",
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
    }

    # Stream B: Attention-Gated MLP
    MLP_PARAMS = {
        "hidden_dim": 256,
        "embedding_dim": 384,  # Must match SBERT_MODEL output
        "dropout": 0.3,  # Dropout for embeddings
        "meta_dropout": 0.1,  # Dropout for metadata branch
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "epochs": 50,  # Sufficient for convergence
        "patience": 15,  # Early stopping patience
    }

    # Ensemble Strategy
    ENSEMBLE_WEIGHTS = {"rf": 0.5, "mlp": 0.5}
