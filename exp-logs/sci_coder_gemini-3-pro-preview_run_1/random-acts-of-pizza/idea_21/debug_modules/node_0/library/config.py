import os
import torch


class Config:
    """
    Centralized configuration for the Hybrid Ensemble project.
    Includes paths, model hyperparameters, and runtime settings.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    RANDOM_SEED = 42

    # ==========================================
    # Paths
    # ==========================================
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for Caching (Read/Write)
    WORKING_DIR = "./working/idea_21"
    CACHE_DIR = WORKING_DIR

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Processing & Feature Engineering
    # ==========================================
    # SBERT
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
    SBERT_DIM = 384

    # TF-IDF
    TFIDF_MAX_FEATURES = 5000
    TFIDF_NGRAM_RANGE = (1, 2)

    # GMM Profiling
    GMM_N_COMPONENTS = 20
    GMM_COVARIANCE_TYPE = "full"

    # ==========================================
    # Model Hyperparameters
    # ==========================================

    # Stream A: Random Forest
    RF_N_ESTIMATORS = 500
    RF_MAX_DEPTH = None  # Full depth
    RF_CLASS_WEIGHT = "balanced"
    RF_N_JOBS = -1

    # Stream B: MLP (Credibility-Gated Attention)
    # Training
    MLP_BATCH_SIZE = 32
    MLP_LEARNING_RATE = 1e-4
    MLP_WEIGHT_DECAY = 1e-4
    MLP_EPOCHS = 50
    MLP_PATIENCE = 15  # Early stopping patience

    # Architecture
    MLP_DROPOUT = 0.3
    MLP_HIDDEN_DIM = 256  # Dimension for internal semantic representations

    # ==========================================
    # Runtime / Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging flag to run on a smaller subset of data
    # Set to True for rapid prototyping, False for full run
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200
