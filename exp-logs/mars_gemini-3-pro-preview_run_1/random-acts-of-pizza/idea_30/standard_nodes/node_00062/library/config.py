import os
import torch


class Config:
    """
    Central configuration for the Hybrid Ensemble solution.
    Handles paths, hyperparameters, and global constants.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_30"
    SUBMISSION_DIR = "./submission"

    # Input Metadata Files (Pre-split)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission Output
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (Deterministic Processing)
    # Stream A: Random Forest Data (Parquet for tabular + TF-IDF features)
    CACHE_RF_TRAIN = os.path.join(WORKING_DIR, "rf_train.parquet")
    CACHE_RF_VAL = os.path.join(WORKING_DIR, "rf_val.parquet")
    CACHE_RF_TEST = os.path.join(WORKING_DIR, "rf_test.parquet")

    # Stream B: MLP Data (NPZ for tensors/embeddings)
    CACHE_MLP_TRAIN = os.path.join(WORKING_DIR, "mlp_train.npz")
    CACHE_MLP_VAL = os.path.join(WORKING_DIR, "mlp_val.npz")
    CACHE_MLP_TEST = os.path.join(WORKING_DIR, "mlp_test.npz")

    # =========================================================================
    # Global Settings
    # =========================================================================
    RANDOM_STATE = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # =========================================================================
    # Feature Engineering (Shared)
    # =========================================================================
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # =========================================================================
    # Stream A: Random Forest Config
    # =========================================================================
    # Model Hyperparameters
    RF_N_ESTIMATORS = 500
    RF_CLASS_WEIGHT = "balanced"
    RF_MIN_SAMPLES_LEAF = 1  # Minimal regularization to preserve sparse Top-K signals
    RF_MAX_DEPTH = None
    RF_N_JOBS = -1

    # TF-IDF Settings (High-Fidelity)
    TFIDF_VOCAB_SIZE = 5000
    TFIDF_NGRAM_RANGE = (1, 2)

    # Community Indicators
    TOP_K_SUBREDDITS = 50

    # =========================================================================
    # Stream B: MLP Config
    # =========================================================================
    # Architecture
    MLP_EMBEDDING_DIM = 384  # Output dimension for all-MiniLM-L6-v2
    MLP_HIDDEN_DIM = 256
    MLP_DROPOUT = 0.3

    # Training
    MLP_LEARNING_RATE = 1e-4
    MLP_WEIGHT_DECAY = 1e-5
    MLP_BATCH_SIZE = 32
    MLP_EPOCHS = 50
    MLP_PATIENCE = 15  # Early stopping patience

    # =========================================================================
    # Ensemble Config
    # =========================================================================
    WEIGHT_RF = 0.5
    WEIGHT_MLP = 0.5

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
