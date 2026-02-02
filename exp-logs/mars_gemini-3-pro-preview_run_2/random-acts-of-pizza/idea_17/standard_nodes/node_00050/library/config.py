import os


class Config:
    """
    Configuration for Idea 17: Asymmetric Multi-View Bagged Linear Ensemble (AMBLE).
    Defines paths, constants, and hyperparameters.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    N_JOBS = 12  # Utilize available vCPUs
    DEVICE = "cuda"  # Use GPU for SBERT encoding

    # ==========================================
    # Directory Structure
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_17"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # File Paths
    # ==========================================
    # Raw Data
    TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

    # Metadata (Splits)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Cache Paths (Intermediate Artifacts)
    # ==========================================
    # Tabular Metadata Features (Parquet)
    TRAIN_TABULAR_PATH = os.path.join(WORKING_DIR, "train_tabular.parquet")
    VAL_TABULAR_PATH = os.path.join(WORKING_DIR, "val_tabular.parquet")
    TEST_TABULAR_PATH = os.path.join(WORKING_DIR, "test_tabular.parquet")

    # View 1: Request Text Embeddings (384d) - Numpy
    TRAIN_REQ_EMB_PATH = os.path.join(WORKING_DIR, "train_req_embeddings.npy")
    VAL_REQ_EMB_PATH = os.path.join(WORKING_DIR, "val_req_embeddings.npy")
    TEST_REQ_EMB_PATH = os.path.join(WORKING_DIR, "test_req_embeddings.npy")

    # View 2: User History Embeddings (Raw 384d, pre-PCA) - Numpy
    TRAIN_HIST_EMB_PATH = os.path.join(WORKING_DIR, "train_hist_embeddings.npy")
    VAL_HIST_EMB_PATH = os.path.join(WORKING_DIR, "val_hist_embeddings.npy")
    TEST_HIST_EMB_PATH = os.path.join(WORKING_DIR, "test_hist_embeddings.npy")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Feature Extraction
    SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    REQ_EMB_DIM = 384
    HIST_EMB_DIM = 384

    # Asymmetric Compression (View 2)
    HISTORY_PCA_COMPONENTS = 16

    # Ensemble Strategy
    N_FOLDS = 5
    BAGGING_N_ESTIMATORS = 20

    # Logistic Regression Grid Search Space
    # High-Regularization Regime, Capped at 10.0
    GRID_SEARCH_PARAMS = {
        "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
        "class_weight": ["balanced", None],
    }
