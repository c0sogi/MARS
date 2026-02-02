import os
import torch


class Config:
    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_18"
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Global Settings
    # ==========================================
    RANDOM_STATE = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Feature Engineering Hyperparameters
    # ==========================================
    # Text Processing (TF-IDF for RF)
    TFIDF_MAX_FEATURES = 5000
    TFIDF_NGRAM_RANGE = (1, 2)

    # Semantic Embeddings (SBERT)
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
    SBERT_EMBEDDING_DIM = 384

    # History Processing
    MAX_HISTORY_LEN = 50  # Max number of subreddits in history sequence for MLP
    PCA_COMPONENTS = 50  # Number of components for compressed history centroids in RF

    # ==========================================
    # Model Hyperparameters: Stream A (Random Forest)
    # ==========================================
    RF_N_ESTIMATORS = 500
    RF_MAX_DEPTH = None
    RF_MIN_SAMPLES_SPLIT = 2
    RF_MIN_SAMPLES_LEAF = 1
    RF_CLASS_WEIGHT = "balanced"
    RF_N_JOBS = -1

    # ==========================================
    # Model Hyperparameters: Stream B (MLP)
    # ==========================================
    # Architecture
    MLP_HIDDEN_DIM = 256
    MLP_DROPOUT = 0.4

    # Training
    MLP_LEARNING_RATE = 1e-4
    MLP_WEIGHT_DECAY = 1e-5
    MLP_BATCH_SIZE = 32
    MLP_EPOCHS = 50
    MLP_PATIENCE = 15  # High patience as requested

    # ==========================================
    # Ensemble Settings
    # ==========================================
    ENSEMBLE_WEIGHTS = {"rf": 0.5, "mlp": 0.5}

    # ==========================================
    # Caching Filenames
    # ==========================================
    # These are stored in WORKING_DIR
    CACHE_FILES = {
        "tfidf_features": "tfidf_features.npz",
        "sbert_request_embeddings": "sbert_request_embeddings.npy",
        "history_sequences": "history_sequences.npy",
        "history_centroids_pca": "history_centroids_pca.npy",
        "metadata_features": "metadata_features.npz",
        "rf_model": "rf_model.joblib",
        "mlp_model": "mlp_model.pth",
    }
