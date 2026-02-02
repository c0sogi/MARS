import os


class Config:
    """
    Global configuration for the H&M Recommendation task.
    """

    # --------------------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # --------------------------------------------------------------------------
    # Input File Paths
    # --------------------------------------------------------------------------
    # Raw metadata
    ARTICLES_PATH = os.path.join(INPUT_DIR, "articles.csv")
    CUSTOMERS_PATH = os.path.join(INPUT_DIR, "customers.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Processed Splits (from metadata generation)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # --------------------------------------------------------------------------
    # Output Paths
    # --------------------------------------------------------------------------
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache directory for this specific idea (Time-Decayed Trend)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    TOP_K = 12  # Number of items to predict per customer
    HISTORY_WEEKS = 5  # Number of weeks of historical transaction data to utilize
    DECAY_ALPHA = (
        2.5  # Constant for time decay formula: score = 1 / (days_elapsed + alpha)
    )

    # --------------------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------------------
    @classmethod
    def create_directories(cls):
        """Ensures that necessary working and submission directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
