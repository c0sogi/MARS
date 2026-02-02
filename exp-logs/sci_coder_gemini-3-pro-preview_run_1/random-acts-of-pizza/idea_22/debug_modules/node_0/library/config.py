import os
import torch


class Config:
    """
    Configuration for the Hybrid Ensemble Pipeline.
    Includes paths, hyperparameters for Random Forest and MLP, and feature engineering settings.
    """

    # ==========================================
    # General & Reproducibility
    # ==========================================
    RANDOM_SEED = 42

    # ==========================================
    # File Paths
    # ==========================================
    # Input Metadata (Pre-generated)
    TRAIN_DATA_PATH = "./metadata/train.csv"
    VAL_DATA_PATH = "./metadata/val.csv"
    TEST_DATA_PATH = "./metadata/test.csv"

    # Working Directory for Caching (Parquet/NPY files)
    WORKING_DIR = "./working/idea_22/"

    # Submission Output
    SUBMISSION_DIR = "./submission/"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Feature Engineering
    # ==========================================
    # Text Embeddings (Sentence-BERT)
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
    SBERT_EMBEDDING_DIM = 384

    # TF-IDF Configuration (Stream A)
    TFIDF_MAX_FEATURES = 5000

    # Target Encoding (Stream A)
    TARGET_ENCODING_FOLDS = 5

    # ==========================================
    # Model A: Random Forest Hyperparameters
    # ==========================================
    RF_N_ESTIMATORS = 500
    RF_CLASS_WEIGHT = "balanced"
    RF_MAX_DEPTH = None  # Allow full depth
    RF_MIN_SAMPLES_LEAF = 4  # Regularization
    RF_N_JOBS = 12  # Utilize available vCPUs

    # ==========================================
    # Model B: Direct-Attention MLP Hyperparameters
    # ==========================================
    # Architecture
    MLP_HIDDEN_DIM = 256
    MLP_DROPOUT = 0.4

    # Training
    MLP_BATCH_SIZE = 32
    MLP_LEARNING_RATE = 1e-4
    MLP_WEIGHT_DECAY = 1e-5
    MLP_EPOCHS = 50
    MLP_PATIENCE = 15  # High patience to allow convergence of attention layers

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Controls for dataset size to speed up development iterations
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200
