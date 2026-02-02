import os
import torch


class Config:
    # =========================================================================
    # 1. Paths and Directories
    # =========================================================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_50"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Input Files (Metadata CSVs)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # 2. Global Settings
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For data loading

    # Debugging flag: If True, runs on a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    # =========================================================================
    # 3. Feature Engineering Configuration
    # =========================================================================
    # Text Embedding Model (Sentence-Transformers)
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

    # TF-IDF Configuration (for Random Forest)
    TFIDF_MAX_FEATURES = 5000
    TFIDF_NGRAM_RANGE = (1, 2)

    # Subreddit History Configuration
    TOP_K_SUBREDDITS = 50  # Number of top frequent subreddits to track as binary flags

    # VADER Sentiment Analysis
    USE_VADER = True

    # =========================================================================
    # 4. Model Hyperparameters: Random Forest (Stream A)
    # =========================================================================
    RF_PARAMS = {
        "n_estimators": 500,
        "class_weight": "balanced",
        "min_samples_leaf": 1,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": 0,
    }

    # =========================================================================
    # 5. Model Hyperparameters: FiLM MLP (Stream B)
    # =========================================================================
    # Architecture Dimensions
    TEXT_EMBED_DIM = 384  # Dimension of all-MiniLM-L6-v2
    FILM_HIDDEN_DIM = 128  # Dimension for the internal FiLM layers
    METADATA_DIM = 0  # Placeholder, calculated dynamically based on input features

    # Training Hyperparameters
    MLP_BATCH_SIZE = 32
    MLP_LEARNING_RATE = 1e-4
    MLP_WEIGHT_DECAY = 1e-5
    MLP_EPOCHS = 50
    MLP_PATIENCE = 15  # Early stopping patience

    # Regularization
    MLP_DROPOUT_EMBED = 0.5  # Dropout on SBERT embeddings
    MLP_DROPOUT_DENSE = 0.2  # Dropout on dense layers

    # Ensemble Weights
    ENSEMBLE_WEIGHT_RF = 0.5
    ENSEMBLE_WEIGHT_MLP = 0.5
