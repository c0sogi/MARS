import os
import torch


class Config:
    """
    Central configuration for the Pizza Request Success Prediction pipeline.
    Includes file paths, hyperparameters for feature engineering, and model settings
    for both the Random Forest and the Orthogonal FiLM-Conditioned MLP.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    # Metadata files generated in the previous step
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw input directory (read-only)
    INPUT_DIR = "./input"
    TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

    # Working directory for caching processed features (Parquet/Numpy)
    # Using specific idea folder to avoid conflicts
    CACHE_DIR = "./working/idea_49/"

    # Output directory for submissions
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # General Settings
    # =========================================================================
    RANDOM_SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging / Development
    # Set DEBUG to True or MAX_SAMPLES to an integer to limit dataset size for fast iteration
    DEBUG = False
    MAX_SAMPLES = None

    # =========================================================================
    # Feature Engineering Hyperparameters
    # =========================================================================
    # Top-K Community Indicators
    TOP_K_SUBREDDITS = 50

    # TF-IDF Configuration for Random Forest
    TFIDF_VOCAB_SIZE = 5000
    TFIDF_NGRAM_RANGE = (1, 2)

    # Semantic Embeddings
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
    MAX_TEXT_LENGTH = 512

    # =========================================================================
    # Model A: Interaction-Augmented Random Forest
    # =========================================================================
    RF_N_ESTIMATORS = 500
    RF_MIN_SAMPLES_LEAF = 1
    RF_CLASS_WEIGHT = "balanced"
    RF_N_JOBS = -1
    RF_RANDOM_STATE = RANDOM_SEED

    # =========================================================================
    # Model B: Orthogonal FiLM-Conditioned MLP
    # =========================================================================
    # Architecture Dimensions
    MLP_EMBEDDING_DIM = 384  # Dimension of all-MiniLM-L6-v2
    MLP_HIDDEN_DIM = 256  # Width of dense layers
    MLP_PROJECTION_DIM = 128  # Dimension for internal projections if needed

    # Regularization
    MLP_DROPOUT_EMB = 0.5  # Dropout for embedding layers
    MLP_DROPOUT_DENSE = 0.2  # Dropout for dense layers

    # Training Loop
    MLP_BATCH_SIZE = 32
    MLP_LEARNING_RATE = 1e-3
    MLP_WEIGHT_DECAY = 1e-4
    MLP_EPOCHS = 50
    MLP_PATIENCE = 15  # Early stopping patience

    # =========================================================================
    # Ensemble Strategy
    # =========================================================================
    # Simple Weighted Average
    ENSEMBLE_WEIGHT_RF = 0.5
    ENSEMBLE_WEIGHT_MLP = 0.5

    @classmethod
    def ensure_dirs(cls):
        """Creates necessary working directories if they don't exist."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
