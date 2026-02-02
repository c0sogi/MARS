import os


class Config:
    """
    Global configuration for the Stratified Directional-Cohort Cascade (SDCC) solution.
    """

    # =========================================================================
    # PATHS
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Dedicated cache directory for Idea 6 (SDCC)
    # Stores intermediate parquet files and numpy matrices
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_6")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary writeable directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # DATA PARAMETERS
    # =========================================================================
    # Number of weeks of transaction history to use for training.
    # Based on "Recency over Volume" principle (5 weeks is often optimal).
    TRAIN_WEEKS = 5

    # Bin size for creating age cohorts (e.g., 18-28, 28-38...)
    # Used for Stratum 3: Cohort-Based Trend
    AGE_BIN_SIZE = 10

    # Minimum purchase count to include an item in the similarity matrix calculation
    # Reduces noise and matrix size
    MIN_ITEM_PURCHASES = 10

    # =========================================================================
    # MODEL HYPERPARAMETERS
    # =========================================================================
    # Weight for the Forward Transition Matrix (S_fwd) in the Hybrid Matrix construction.
    # S_hybrid = S_sym + (LAMBDA_FWD * S_fwd)
    # Higher value emphasizes directional "next-item" patterns over basket co-occurrence.
    LAMBDA_FWD = 0.5

    # Power parameter for time decay weighting: weight = 1 / (days_elapsed + 1) ** ALPHA
    TIME_DECAY_ALPHA = 2.5

    # Number of predictions per customer
    TOP_K = 12

    # =========================================================================
    # STRATIFICATION OFFSETS (THE CASCADE)
    # =========================================================================
    # These offsets enforce the strict hierarchy of signal sources.
    # Logic: Score = Raw_Score + Offset
    # Hierarchy: Habitual Repurchase > Directional CF > Cohort Trend > Global Trend

    OFFSET_HISTORY = 1000.0  # Stratum 1: High precision priors
    OFFSET_CF = 100.0  # Stratum 2: Personalized discovery
    OFFSET_COHORT = 10.0  # Stratum 3: Demographic fallback
    OFFSET_GLOBAL = 0.0  # Stratum 4: Universal fallback

    # =========================================================================
    # REPRODUCIBILITY
    # =========================================================================
    SEED = 42
