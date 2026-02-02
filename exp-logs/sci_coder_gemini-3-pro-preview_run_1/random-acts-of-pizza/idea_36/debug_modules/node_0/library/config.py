import os
import torch


class Config:
    # ==========================================
    # Global Settings
    # ==========================================
    RANDOM_STATE = 42
    DEBUG = False  # Set to True to run on a subset of data for debugging
    DEBUG_SAMPLE_SIZE = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # For data loading

    # ==========================================
    # Directory & File Paths
    # ==========================================
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching (Write Allowed)
    # Using specific subdirectory for this idea to avoid conflicts
    WORKING_DIR = "./working/idea_36"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache file paths
    CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features.parquet")
    CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "val_features.parquet")
    CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "test_features.parquet")

    # Embeddings cache
    CACHE_SBERT_EMBEDDINGS = os.path.join(WORKING_DIR, "sbert_embeddings.npz")

    # ==========================================
    # Feature Engineering Hyperparameters
    # ==========================================
    # Text Processing
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    MAX_TEXT_LENGTH = 512  # Token limit for SBERT

    # TF-IDF (Stream A)
    TFIDF_VOCAB_SIZE = 5000
    TFIDF_NGRAM_RANGE = (1, 2)

    # Top-K Subreddits
    TOP_K_SUBREDDITS = 50

    # Metadata Handling
    USE_ARCSINH_TRANSFORM = True

    # ==========================================
    # Model Hyperparameters: Random Forest (Stream A)
    # ==========================================
    RF_N_ESTIMATORS = 500
    RF_MAX_DEPTH = None
    RF_MIN_SAMPLES_LEAF = 1  # Low regularization to preserve sparse signals
    RF_CLASS_WEIGHT = "balanced"
    RF_N_JOBS = -1

    # ==========================================
    # Model Hyperparameters: MLP (Stream B)
    # ==========================================
    # Architecture
    MLP_HIDDEN_DIMS = [256, 128]
    MLP_DROPOUT_EMB = 0.5  # High dropout on embeddings
    MLP_DROPOUT_DENSE = 0.2

    # Training
    MLP_BATCH_SIZE = 32
    MLP_LEARNING_RATE = 1e-4
    MLP_WEIGHT_DECAY = 1e-4
    MLP_EPOCHS = 50
    MLP_PATIENCE = 15  # High patience for early stopping

    # Attention Mechanism
    ATTENTION_HEADS = 4  # If using MultiheadAttention

    # ==========================================
    # Ensemble Settings
    # ==========================================
    ENSEMBLE_WEIGHT_RF = 0.5
    ENSEMBLE_WEIGHT_MLP = 0.5

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("-" * 30)
        print("Configuration:")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Random State: {cls.RANDOM_STATE}")
        print(f"  Debug Mode: {cls.DEBUG}")
        print(f"  Working Dir: {cls.WORKING_DIR}")
        print(f"  SBERT Model: {cls.SBERT_MODEL_NAME}")
        print(f"  RF Estimators: {cls.RF_N_ESTIMATORS}")
        print(f"  MLP Hidden Dims: {cls.MLP_HIDDEN_DIMS}")
        print("-" * 30)
