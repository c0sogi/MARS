import os


class Config:
    # --------------------------------------------------------------------------
    # Global Seeding
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # Directory and File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    CACHE_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Input Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Artifacts
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    TFIDF_VECTORIZER_PATH = os.path.join(WORKING_DIR, "tfidf_vectorizer.joblib")
    RIDGE_MODEL_PATH = os.path.join(WORKING_DIR, "ridge_model.joblib")

    # --------------------------------------------------------------------------
    # Data Processing & Feature Engineering
    # --------------------------------------------------------------------------
    # Sparse Stream (TF-IDF)
    VOCAB_SIZE = 60000

    # Dense Stream (Transformer)
    TRANSFORMER_MODEL_NAME = "microsoft/codebert-base"
    MAX_LEN = 512

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.01
    EPOCHS = 2
    FP16 = True
    NUM_WORKERS = 4

    # --------------------------------------------------------------------------
    # Ensemble Strategy
    # --------------------------------------------------------------------------
    # Weight for the Sparse (Ridge) prediction.
    # Final Rank = ALPHA * Ridge_Rank + (1 - ALPHA) * Transformer_Rank
    ALPHA = 0.5

    # --------------------------------------------------------------------------
    # Debugging & Development
    # --------------------------------------------------------------------------
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000  # Number of samples to use when DEBUG is True

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories if they do not exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
