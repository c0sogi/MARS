import os
import torch


class Config:
    # =========================================
    # Global Settings
    # =========================================
    RANDOM_STATE = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For data loading

    # =========================================
    # File Paths & Directories
    # =========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Data Paths (Metadata CSVs)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Original JSON Paths (for reference or deep text extraction if needed)
    TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths
    # We use .parquet for dataframes and .npy/.pt for tensors/embeddings
    CACHE_TRAIN_PROCESSED = os.path.join(WORKING_DIR, "train_processed.parquet")
    CACHE_VAL_PROCESSED = os.path.join(WORKING_DIR, "val_processed.parquet")
    CACHE_TEST_PROCESSED = os.path.join(WORKING_DIR, "test_processed.parquet")

    CACHE_EMBEDDINGS = os.path.join(WORKING_DIR, "sbert_embeddings.pt")
    CACHE_TOPIC_MODEL = os.path.join(WORKING_DIR, "topic_model.pkl")

    # =========================================
    # Feature Engineering Config
    # =========================================
    # Text Columns
    TEXT_COL_TITLE = "request_title"
    TEXT_COL_BODY = "request_text_edit_aware"

    # Subreddit History Column
    SUBREDDIT_LIST_COL = "requester_subreddits_at_request"
    MAX_HISTORY_LENGTH = (
        20  # Max number of subreddits to consider for attention sequence
    )

    # Topic Modeling (Stream A & B Pre-processing)
    NUM_TOPICS = 15
    TOPIC_MODEL_TYPE = "NMF"  # Non-Negative Matrix Factorization

    # Sentence Transformer
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    # Domain Lexicons (for Density Features)
    LEXICONS = {
        "reciprocity": [
            "return",
            "pay",
            "back",
            "repay",
            "favor",
            "forward",
            "check",
            "friday",
            "tomorrow",
            "paid",
        ],
        "desperation": [
            "broke",
            "hungry",
            "food",
            "money",
            "job",
            "rent",
            "starving",
            "empty",
            "last",
            "days",
            "week",
        ],
        "gratitude": [
            "thanks",
            "thank",
            "appreciate",
            "grateful",
            "love",
            "awesome",
            "bless",
        ],
    }

    # Metadata Columns to use (Numerical)
    # These will be Arcsinh transformed for MLP and used raw for RF
    NUMERIC_META_COLS = [
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

    # =========================================
    # Model Hyperparameters
    # =========================================

    # Stream A: Random Forest
    RF_PARAMS = {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    # Stream B: Credibility-Gated Attention MLP
    MLP_PARAMS = {
        "hidden_dim": 256,
        "dropout_rate": 0.3,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 50,
        "patience": 15,  # Early stopping patience
        "scheduler_factor": 0.5,
        "scheduler_patience": 5,
    }

    # Ensemble Weights
    ENSEMBLE_WEIGHTS = {"rf": 0.5, "mlp": 0.5}
