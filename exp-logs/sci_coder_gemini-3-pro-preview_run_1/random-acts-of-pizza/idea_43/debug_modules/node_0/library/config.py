import os
import torch


class Config:
    """
    Configuration for the Hybrid Ensemble solution.
    Includes settings for Paths, Feature Engineering, Random Forest, and MLP.
    """

    # --------------------------------------------------------------------------
    # Global Seeding & Debugging
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SUBSET_SIZE = 100

    # --------------------------------------------------------------------------
    # File Paths & Directories
    # --------------------------------------------------------------------------
    # Input Metadata (Generated in previous steps)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Caching (Parquet/Numpy files)
    # Stores intermediate processed features to save time on re-runs
    WORKING_DIR = "./working/idea_43"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Output Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Feature Engineering Hyperparameters
    # --------------------------------------------------------------------------
    # Text Processing
    SBERT_MODEL_NAME = (
        "all-MiniLM-L6-v2"  # 384-dimensional embeddings, fast and effective
    )
    TFIDF_MAX_FEATURES = 5000  # Vocabulary size for TF-IDF

    # Community & History
    TOP_K_SUBREDDITS = 50  # Number of most frequent subreddits to track as binary flags

    # Key Columns
    TEXT_COL_TITLE = "request_title"
    TEXT_COL_BODY = "request_text_edit_aware"
    TARGET_COL = "requester_received_pizza"
    ID_COL = "request_id"

    # --------------------------------------------------------------------------
    # Model Stream A: Random Forest (Consistency-Augmented Top-K)
    # --------------------------------------------------------------------------
    RF_PARAMS = {
        "n_estimators": 500,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,  # Low regularization to preserve sparse signals from Top-K
        "min_weight_fraction_leaf": 0.0,
        "max_features": "sqrt",
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "bootstrap": True,
        "oob_score": False,
        "n_jobs": -1,  # Use all available cores
        "random_state": SEED,
        "verbose": 0,
        "warm_start": False,
        "class_weight": "balanced",  # Handle the ~25% positive class imbalance
        "ccp_alpha": 0.0,
        "max_samples": None,
    }

    # --------------------------------------------------------------------------
    # Model Stream B: Unified Credibility-Gated MLP
    # --------------------------------------------------------------------------
    # Architecture
    EMBEDDING_DIM = 384  # Dimension of SBERT embeddings (MiniLM)
    HIDDEN_DIM = 256  # Dimension of internal dense layers
    PROJECTION_DIM = 128  # Dimension for projected spaces (if used)

    # Regularization
    DROPOUT_EMB = (
        0.5  # High dropout on embeddings to prevent overfitting to specific phrasing
    )
    DROPOUT_DENSE = 0.2  # Standard dropout on dense layers

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    PATIENCE = 15  # Early stopping patience to allow stabilization of attention layers

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    # --------------------------------------------------------------------------
    # Ensemble Strategy
    # --------------------------------------------------------------------------
    ENSEMBLE_WEIGHTS = {"rf": 0.5, "mlp": 0.5}
