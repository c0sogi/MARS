import os


class Config:
    """
    Configuration class for the Toxicity Classification task.
    Centralizes hyperparameters, file paths, and runtime settings for
    Ridge Regression with Bias-Corrective Resampling.
    """

    # ==========================================
    # Random Seed & Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # File Paths
    # ==========================================
    # Input Metadata (Generated in previous step, Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    # working/idea_1 is used for caching intermediate files (e.g. vectorizers)
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing & Vectorization
    # ==========================================
    # TF-IDF Word Vectorizer Settings
    # Unigrams and Bigrams to capture phrases
    WORD_NGRAM_RANGE = (1, 2)
    WORD_MAX_FEATURES = 150000  # High dimensionality for linear models
    WORD_MIN_DF = 3  # Ignore terms appearing in fewer than 3 docs

    # TF-IDF Character Vectorizer Settings
    # 2-5 character n-grams to capture subword information and obfuscations
    CHAR_NGRAM_RANGE = (2, 5)
    CHAR_MAX_FEATURES = 100000
    CHAR_MIN_DF = 3

    # ==========================================
    # Bias Mitigation Strategy (Resampling)
    # ==========================================
    # We perform Stratified Oversampling on examples that mention identities.
    # This weight determines how many times we duplicate specific subgroup rows
    # (Toxic+Identity and Non-Toxic+Identity) to increase their influence on the loss.
    RESAMPLE_WEIGHT = 3

    # ==========================================
    # Model Hyperparameters (Ridge Regression)
    # ==========================================
    # Regularization strength (alpha).
    # Optimization minimizes: ||y - Xw||^2_2 + alpha * ||w||^2_2
    RIDGE_ALPHA = 1.0

    # Solver for Ridge Regression ('auto' usually selects 'lsqr' or 'sag' for sparse data)
    RIDGE_SOLVER = "auto"

    # ==========================================
    # Runtime & Debugging
    # ==========================================
    # Set DEBUG to True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 20000

    def __init__(self):
        """
        Initialize configuration and ensure necessary directories exist.
        """
        # Ensure output directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)
