import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Reproducibility & Debugging
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing
    DEBUG_SIZE = 100  # Number of samples to use when DEBUG is True

    # --------------------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed features (Stream A & B data)
    WORKING_DIR = "./working/idea_9"

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Data Columns
    # --------------------------------------------------------------------------
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    # Text Inputs
    TEXT_COL = "request_text_edit_aware"  # Preferred edit-aware text
    TITLE_COL = "request_title"

    # Feature Engineering Inputs
    SUBREDDIT_LIST_COL = "requester_subreddits_at_request"

    # Columns to exclude from tabular features (IDs, raw text, leakage, etc.)
    # Note: 'at_retrieval' leakage columns are typically handled by
    # intersecting train/test columns in the pipeline, but we list explicit exclusions here.
    DROP_COLS = [
        "request_id",
        "giver_username_if_known",
        "source_file",
        "request_text",
        "request_text_edit_aware",
        "request_title",
        "requester_subreddits_at_request",
        "requester_username",
        "requester_user_flair",
        "requester_received_pizza",
    ]

    # --------------------------------------------------------------------------
    # Feature Engineering Configurations
    # --------------------------------------------------------------------------
    # Stream A: TF-IDF Settings
    TFIDF_MAX_FEATURES = 5000
    TFIDF_NGRAM_RANGE = (1, 2)

    # Stream B: SBERT & Attention Settings
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
    SBERT_EMBEDDING_DIM = 384
    MAX_SUBREDDIT_SEQ_LEN = (
        20  # Max number of historical subreddits to process via attention
    )

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    # Stream A: Augmented Random Forest
    RF_PARAMS = {
        "n_estimators": 500,
        "max_depth": None,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1,
    }

    # Stream B: Attention-Gated MLP
    MLP_PARAMS = {
        "hidden_dim": 256,
        "attention_dim": 128,
        "dropout_rate": 0.3,
        "metadata_dropout_rate": 0.0,  # Low dropout for dense metadata features
    }

    # --------------------------------------------------------------------------
    # Training Configuration
    # --------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 50
    PATIENCE = 15  # Early stopping patience

    # --------------------------------------------------------------------------
    # Ensemble Configuration
    # --------------------------------------------------------------------------
    # Simple weighted average
    ENSEMBLE_WEIGHTS = {"rf": 0.5, "mlp": 0.5}
