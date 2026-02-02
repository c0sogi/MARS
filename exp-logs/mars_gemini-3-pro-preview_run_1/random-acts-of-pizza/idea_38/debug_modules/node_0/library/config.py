import os


class Config:
    """
    Configuration class for the Hybrid Ensemble with Semantic Prototype Profiling
    and Dual-Query Alignment pipeline.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    RANDOM_SEED = 42

    # Debugging flag to run on a smaller subset of data for quick testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    # ==========================================
    # File Paths
    # ==========================================
    # Input Data (Metadata)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_38"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Feature Engineering Parameters
    # ==========================================
    # Sentence-BERT model for semantic embeddings
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

    # Number of top frequent subreddits to encode as binary flags
    TOP_K_SUBREDDITS = 50

    # Vocabulary size for TF-IDF vectorization (Stream A)
    TFIDF_VOCAB_SIZE = 5000

    # Text columns to process
    TEXT_COL_TITLE = "request_title"
    TEXT_COL_BODY = "request_text_edit_aware"

    # ==========================================
    # Model Hyperparameters: Random Forest (Stream A)
    # ==========================================
    RF_PARAMS = {
        "n_estimators": 500,
        "class_weight": "balanced",
        "min_samples_leaf": 1,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbose": 0,
    }

    # ==========================================
    # Model Hyperparameters: MLP (Stream B)
    # ==========================================
    MLP_PARAMS = {
        "hidden_dim": 256,
        "dropout_emb": 0.5,  # Dropout applied to embeddings
        "dropout_dense": 0.2,  # Dropout applied to dense layers
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "batch_size": 32,
        "epochs": 50,
        "patience": 15,  # Early stopping patience
        "device": "cuda",  # Will fallback to cpu in code if not available
    }

    # ==========================================
    # Ensemble Configuration
    # ==========================================
    # Weights for the final simple weighted average
    ENSEMBLE_WEIGHT_RF = 0.5
    ENSEMBLE_WEIGHT_MLP = 0.5
