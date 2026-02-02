import os


class Config:
    """
    Configuration class for the Frozen Transformer + Ridge Regression pipeline.
    """

    # =========================================
    # Global Settings
    # =========================================
    SEED = 42

    # =========================================
    # File Paths
    # =========================================
    # Input Metadata directories (Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Working Directory for Intermediate Artifacts (Cache, Models)
    WORKING_DIR = "./working/idea_1"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_DIR = os.path.join(WORKING_DIR, "models")

    # =========================================
    # Model Hyperparameters
    # =========================================
    # Pre-trained Sentence Transformer model name
    # We use a lightweight model for efficiency as per the task description
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    # Maximum sequence length for the transformer
    MAX_LENGTH = 512

    # Batch size for generating embeddings (Inference)
    BATCH_SIZE = 64

    # Ridge Regression Hyperparameters
    # List of alpha (regularization strength) values to try during Cross-Validation
    RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    # =========================================
    # Execution Control
    # =========================================
    # Number of CPU workers for data loaders (where applicable)
    NUM_WORKERS = 4

    # Debugging flags
    # If True, the pipeline will only process a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200

    @staticmethod
    def setup():
        """
        Ensures necessary working directories exist.
        """
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
