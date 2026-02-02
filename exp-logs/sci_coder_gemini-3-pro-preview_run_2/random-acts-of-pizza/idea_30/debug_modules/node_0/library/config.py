import os
import numpy as np
import random


class Config:
    """
    Global configuration for the Asymmetric Dual-Backbone Consensus (ADBC) strategy.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLES = 100  # Number of samples to use when DEBUG is True

    # ==========================================
    # Directory & File Paths
    # ==========================================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sampleSubmission.csv")

    # Metadata Directories (Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Working / Cache Directory (Write Allowed)
    WORKING_DIR = "./working/idea_30"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Cache File Paths
    # ==========================================
    # Embeddings (Numpy format)
    TRAIN_EMBEDDINGS_PRIMARY = os.path.join(WORKING_DIR, "train_embeddings_primary.npy")
    TRAIN_EMBEDDINGS_AUX = os.path.join(WORKING_DIR, "train_embeddings_aux.npy")

    VAL_EMBEDDINGS_PRIMARY = os.path.join(WORKING_DIR, "val_embeddings_primary.npy")
    VAL_EMBEDDINGS_AUX = os.path.join(WORKING_DIR, "val_embeddings_aux.npy")

    TEST_EMBEDDINGS_PRIMARY = os.path.join(WORKING_DIR, "test_embeddings_primary.npy")
    TEST_EMBEDDINGS_AUX = os.path.join(WORKING_DIR, "test_embeddings_aux.npy")

    # Processed Features (Parquet format)
    TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES = os.path.join(WORKING_DIR, "test_features.parquet")

    # ==========================================
    # Model Architecture Configuration
    # ==========================================
    # Primary Backbone: High-Resolution Anchor (Frozen)
    PRIMARY_BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"

    # Auxiliary Backbone: Low-Resolution World Knowledge (Frozen)
    AUX_BACKBONE = "sentence-transformers/all-mpnet-base-v2"

    # Dimensionality Reduction for Auxiliary View
    # We compress the 768d MPNet embeddings to 50d via PCA
    AUX_PCA_COMPONENTS = 50

    # ==========================================
    # Feature Selection
    # ==========================================
    # Text Inputs for Embedding Generation
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Robust Metadata Features (~10 dimensions)
    # Selected based on stability and relevance, excluding explicit user history lists
    METADATA_COLS = [
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
    # Training Hyperparameters
    # ==========================================
    # Cross-Validation
    N_FOLDS = 5

    # Ensemble Strategy
    N_BAGGING_ESTIMATORS = 20  # Number of base estimators in BaggingClassifier

    # Logistic Regression Base Learner Settings
    # Search Space for Grid Search
    LR_C_RANGE = np.logspace(-4, 1, 20)  # 1e-4 to 10.0
    LR_PENALTY = "l2"
    LR_SOLVER = "lbfgs"
    LR_CLASS_WEIGHTS = ["balanced", None]
    LR_MAX_ITER = 1000


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and Torch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
