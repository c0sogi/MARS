import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # 1. Global & System Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEBUG = False  # Set to True to run on a small subset for testing
    DEBUG_SAMPLE_SIZE = 100  # Number of samples if DEBUG is True

    # ==========================================
    # 2. File Paths
    # ==========================================
    # Input Metadata (Pre-generated)
    TRAIN_DATA_PATH = "./metadata/train.csv"
    VAL_DATA_PATH = "./metadata/val.csv"
    TEST_DATA_PATH = "./metadata/test.csv"

    # Working Directory for Caching (Deterministic Processing)
    CACHE_DIR = "./working/idea_55/"

    # Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 3. Feature Engineering Hyperparameters
    # ==========================================
    # Text Embeddings
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
    SBERT_EMBEDDING_DIM = 384

    # TF-IDF (for Random Forest)
    TFIDF_MAX_FEATURES = 5000

    # Community Interaction
    TOP_K_SUBREDDITS = 50

    # ==========================================
    # 4. Model Hyperparameters
    # ==========================================
    # Stream A: Interaction-Enhanced Consistency Random Forest
    RF_N_ESTIMATORS = 500
    RF_MIN_SAMPLES_LEAF = 1
    RF_CLASS_WEIGHT = "balanced"
    RF_RANDOM_STATE = SEED

    # Stream B: Orthogonal Skip-Gated MLP
    MLP_HIDDEN_DIM = 256
    MLP_DROPOUT_EMB = 0.5  # Higher dropout for embeddings
    MLP_DROPOUT_DENSE = 0.2  # Lower dropout for internal dense layers

    # MLP Training
    MLP_LEARNING_RATE = 1e-4
    MLP_WEIGHT_DECAY = 1e-2
    MLP_BATCH_SIZE = 32
    MLP_EPOCHS = 50
    MLP_PATIENCE = 15  # High patience for stability

    # ==========================================
    # 5. Ensemble Settings
    # ==========================================
    # Weights for [RandomForest, MLP]
    ENSEMBLE_WEIGHTS = (0.5, 0.5)

    # ==========================================
    # 6. Column Definitions
    # ==========================================
    # Target
    TARGET_COL = "requester_received_pizza"

    # Text Columns
    TEXT_COLS = {
        "title": "request_title",
        "body": "request_text_edit_aware",  # Use edit-aware to prevent leakage
    }

    # Numerical Metadata Columns (to be Arcsinh transformed for MLP, raw for RF)
    NUMERICAL_COLS = [
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        # Note: We exclude retrieval-time features to prevent leakage if they weren't masked,
        # but the dataset description implies 'at_request' cols are safe.
    ]

    # List columns (for history processing)
    LIST_COLS = {"subreddits": "requester_subreddits_at_request"}


def setup_reproducibility(seed=Config.SEED):
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create necessary directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
