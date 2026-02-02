import os


class Config:
    """
    Configuration class for the Whitened Multi-Field Asymmetric Dual-Backbone Ensemble.
    Defines file paths, model constants, and training hyperparameters.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    # Debugging flags to control dataset size
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    # ==========================================
    # Directory Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea iteration
    WORKING_DIR = "./working/idea_41"
    SUBMISSION_DIR = "./submission"

    # ==========================================
    # File Paths
    # ==========================================
    # Raw Data
    TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

    # Metadata
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Caching Paths (in Working Directory)
    # ==========================================
    # Embeddings
    TRAIN_ANCHOR_EMB_PATH = os.path.join(WORKING_DIR, "train_anchor_embeddings.npy")
    TRAIN_AUX_EMB_PATH = os.path.join(WORKING_DIR, "train_aux_embeddings.npy")
    TEST_ANCHOR_EMB_PATH = os.path.join(WORKING_DIR, "test_anchor_embeddings.npy")
    TEST_AUX_EMB_PATH = os.path.join(WORKING_DIR, "test_aux_embeddings.npy")

    # Processed Features
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # ==========================================
    # Model Architecture Constants
    # ==========================================
    # Field-Specific Anchor Backbone (High-Resolution, 384d)
    ANCHOR_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    # Global Auxiliary Backbone (High-Capacity, 768d)
    AUX_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

    # ==========================================
    # Feature Engineering Hyperparameters
    # ==========================================
    # PCA Whitening for Auxiliary View
    PCA_N_COMPONENTS = 50
    PCA_WHITEN = True

    # Metadata Processing
    SCALER_OUTPUT_DISTRIBUTION = "normal"  # For QuantileTransformer (RankGauss)

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    N_FOLDS = 5

    # Ensemble Settings
    N_BAGGING_ESTIMATORS = 50

    # Logistic Regression Grid Search Space
    # Keys must match the pipeline step names if used in a pipeline,
    # or be passed to the estimator directly.
    LR_PARAM_GRID = {
        "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "class_weight": ["balanced", None],
        "solver": ["lbfgs"],
        "max_iter": [1000],
    }

    @classmethod
    def setup(cls):
        """
        Creates the necessary working and submission directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
