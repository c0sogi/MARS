import os


class Config:
    """
    Configuration for the Inventory-Gated Dual-Window Cascade (IGDC) model.
    Defines file paths, hyperparameters, temporal windows, and stratification logic.
    """

    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a subset of data for debugging
    DEBUG_ROWS = 100_000  # Number of rows to load if DEBUG is True

    # =========================================================================
    # Directory Paths
    # =========================================================================
    # Input data (Read-Only)
    INPUT_DIR = "./input"

    # Pre-generated Metadata (Read-Only)
    METADATA_DIR = "./metadata"

    # Working Directory for Caching (Read/Write)
    # Using 'idea_14' as specified for this solution
    WORKING_DIR = "./working/idea_14"

    # Submission Output Directory
    SUBMISSION_DIR = "./submission"

    # =========================================================================
    # File Paths
    # =========================================================================
    # Raw/Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    ARTICLES_CSV = os.path.join(INPUT_DIR, "articles.csv")
    CUSTOMERS_CSV = os.path.join(INPUT_DIR, "customers.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Submission File
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Caching Paths (Parquet/Numpy)
    # =========================================================================
    # These files will be created in WORKING_DIR if they don't exist
    CACHE_SIMILARITY_MATRIX = os.path.join(WORKING_DIR, "similarity_matrix.npz")
    CACHE_USER_HISTORY = os.path.join(WORKING_DIR, "user_history.parquet")
    CACHE_ITEM_MAP = os.path.join(WORKING_DIR, "item_map.parquet")
    CACHE_USER_MAP = os.path.join(WORKING_DIR, "user_map.parquet")
    CACHE_INVENTORY_MASK = os.path.join(WORKING_DIR, "inventory_mask.npy")
    CACHE_GLOBAL_TREND = os.path.join(WORKING_DIR, "global_trend.npy")
    CACHE_HABIT_MATRIX = os.path.join(WORKING_DIR, "habit_matrix.npz")

    # =========================================================================
    # Model Hyperparameters & Temporal Windows
    # =========================================================================
    # The reference date is the last day of the training data (2020-09-22)
    REFERENCE_DATE = "2020-09-22"

    # Temporal Windows (in days)
    # Structure Learning (S_long): 16 weeks to capture long-tail correlations
    WINDOW_STRUCTURE_DAYS = 16 * 7  # 112 days

    # Intent Inference (U_short): 2 weeks to capture immediate user context
    WINDOW_INTENT_DAYS = 2 * 7  # 14 days

    # Habitual Repurchase: 4 weeks (Priors Layer)
    WINDOW_HABIT_DAYS = 4 * 7  # 28 days

    # Inventory/Trend: 1 week (Feasibility & Fallback)
    WINDOW_INVENTORY_DAYS = 7  # 7 days

    # Collaborative Filtering Parameters
    TOP_K_NEIGHBORS = 100  # Pruning size for Item-Item Similarity matrix
    SHRINKAGE = 0  # Optional shrinkage for similarity calculation

    # Prediction Parameters
    TOP_K_PREDICTIONS = 12  # Number of items to predict per user

    # =========================================================================
    # Stratification & Scoring Logic
    # =========================================================================
    # The model produces a unified score vector R_total.
    # Ranges are disjoint to enforce the hierarchy: Habit > CF > Trend.

    # Stratum 1: Habitual Repurchase (The "Priors" Layer)
    # Scores are shifted to be > 2000.
    SCORE_HABIT_OFFSET = 2000.0

    # Stratum 2: Inventory-Gated CF (The "Discovery" Layer)
    # Normalized scores are scaled to fit within [100, 1000].
    SCORE_CF_MIN = 100.0
    SCORE_CF_MAX = 1000.0

    # Stratum 3: Global Trend (The "Fallback" Layer)
    # Scores are scaled to fit within [0, 10].
    SCORE_TREND_MAX = 10.0

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
