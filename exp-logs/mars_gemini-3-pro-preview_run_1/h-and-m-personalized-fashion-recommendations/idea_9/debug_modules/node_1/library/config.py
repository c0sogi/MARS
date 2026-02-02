import os


class Config:
    """
    Global configuration for the Decay-Weighted Stratified Cascade (DWSC) model.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for Idea 9 (DWSC) specific artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_9")
    SUBMISSION_DIR = "./submission"

    # Input Files
    # Generated via metadata script:
    # train.csv: 80% of users with full history
    # val.csv: 20% of users with full history (used for validation split)
    # test.csv: Users requiring prediction in submission
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    ARTICLES_PATH = os.path.join(INPUT_DIR, "articles.csv")
    CUSTOMERS_PATH = os.path.join(INPUT_DIR, "customers.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 12  # Matches available vCPUs

    # Debugging / Development
    # Set DEBUG to True to load a smaller subset of data for rapid iteration
    DEBUG = False
    DEBUG_ROWS = 100000

    # =========================================================================
    # Model Hyperparameters (Decay-Weighted Stratified Cascade)
    # =========================================================================

    # 1. Temporal Windowing
    # We use a 10-week window for training the CF matrix and history retrieval.
    # This balances the "Long Tail" (volume) with "Recency" (relevance).
    TRAIN_WINDOW_WEEKS = 10

    # 2. Stratum 1: Habitual Repurchase (The "Priors" Layer)
    # Logic: Explicit history always outranks inferred similarity.
    # Scores are shifted to [HISTORY_OFFSET, inf).
    HISTORY_OFFSET = 2000.0

    # Decay function: weight = 1 / (days_elapsed + 1)^power
    # Power 1.0 implies strict hyperbolic decay (1/t).
    HISTORY_DECAY_POWER = 1.0

    # 3. Stratum 2: Decay-Weighted Collaborative Filtering (The "Discovery" Layer)
    # Logic: Inferred similarity outranks global trends but not explicit history.
    # Scores are normalized and scaled to [CF_OFFSET_MIN, CF_OFFSET_MAX].
    CF_OFFSET_MIN = 100.0
    CF_OFFSET_MAX = 1000.0

    # Decay for Interaction Matrix construction: weight = 1 / (days_elapsed + 1)^power
    # Power 0.5 implies 1/sqrt(t), which is gentler than history decay,
    # allowing older co-occurrences to still contribute to the manifold.
    CF_DECAY_POWER = 0.5

    # Pruning: Number of neighbors to retain per item in the sparse Similarity Matrix.
    CF_NEIGHBORS = 100

    # 4. Stratum 3: Global Trend (The "Fallback" Layer)
    # Logic: Fallback for cold-start or low-confidence predictions.
    # Scores are scaled to [0, TREND_OFFSET_MAX].
    TREND_OFFSET_MAX = 10.0

    # =========================================================================
    # Inference Settings
    # =========================================================================
    TOP_K = 12
    # Batch size for vectorized user processing/inference
    BATCH_SIZE = 5000
    # Precision for matrix operations (float32 required for stability with offsets)
    PRECISION = "float32"

    @classmethod
    def setup(cls):
        """
        Ensures necessary working directories exist.
        Should be called at the start of execution.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
