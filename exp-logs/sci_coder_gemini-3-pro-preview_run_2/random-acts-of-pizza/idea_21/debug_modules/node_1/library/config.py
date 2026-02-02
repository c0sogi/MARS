import os


class Config:
    """
    Configuration class for the Topic-Augmented Dense Fusion (TADF) strategy.
    Defines paths, hyperparameters, and global constants.
    """

    # --------------------------------------------------------------------------
    # Global Settings
    # --------------------------------------------------------------------------
    RANDOM_SEED = 42
    # Debugging flags to control dataset size for rapid testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    # --------------------------------------------------------------------------
    # Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_21"
    SUBMISSION_DIR = "./submission"

    # --------------------------------------------------------------------------
    # Input Data Paths
    # --------------------------------------------------------------------------
    # Raw JSON files
    TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

    # Generated Metadata CSVs
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # --------------------------------------------------------------------------
    # Output Paths
    # --------------------------------------------------------------------------
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Caching Paths (for deterministic processing)
    # --------------------------------------------------------------------------
    # SBERT Embeddings (Numpy arrays)
    TRAIN_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "train_embeddings.npy")
    VAL_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "val_embeddings.npy")
    TEST_EMBEDDINGS_PATH = os.path.join(WORKING_DIR, "test_embeddings.npy")

    # Processed Features (Parquet files containing metadata & text)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    # Text Backbone
    SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    # User History Backbone (LDA)
    LDA_N_COMPONENTS = 10
    LDA_RANDOM_STATE = RANDOM_SEED
    # Minimum document frequency for CountVectorizer in LDA pipeline
    LDA_MIN_DF = 5

    # Classifier (Logistic Regression)
    # Hyperparameter Grid for optimization
    PARAM_GRID = {
        "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "class_weight": ["balanced", None],
    }

    # Ensemble (Bagging)
    BAGGING_N_ESTIMATORS = 20

    # Training Configuration
    N_FOLDS = 5

    @classmethod
    def ensure_directories(cls):
        """
        Creates necessary working and submission directories if they don't exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
