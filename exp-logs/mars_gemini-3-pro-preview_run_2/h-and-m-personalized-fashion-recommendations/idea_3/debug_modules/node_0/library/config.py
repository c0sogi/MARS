import os
import numpy as np
from pathlib import Path


class Config:
    """
    Configuration for Hybrid Multi-Source Retrieval & Interaction-Aware Ranking.
    Centralizes paths, hyperparameters, and model settings.
    """

    # -------------------------------------------------------------------------
    # Global Configuration
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    # Read-only input directories
    INPUT_DIR = Path("./input")
    META_DIR = Path("./metadata")

    # Writeable working directory for caching and intermediate files
    # We use 'idea_3' to isolate this iteration's artifacts
    WORKING_DIR = Path("./working/idea_3")

    # Output directory for final submission
    SUBMISSION_DIR = Path("./submission")

    # Ensure writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Split & Time Windows
    # -------------------------------------------------------------------------
    # Number of days to use for validation (the last days of training data)
    # Strategy: Time-based split to mimic test scenario
    VAL_DAYS = 7

    # Number of weeks of history to use for the Retrieval Stage (Candidate Generation)
    # Strategy: Restrict to recent 4 weeks to capture current trends and reduce noise
    RETRIEVAL_HISTORY_WEEKS = 4

    # -------------------------------------------------------------------------
    # Retrieval (Candidate Generation) Hyperparameters
    # -------------------------------------------------------------------------
    # Source A: Item-Item Co-occurrence
    # Decay function for co-occurrence weights: 'linear' (w = 1/t)
    COOC_DECAY_MODE = "linear"
    # Do not normalize co-occurrence scores (keep raw popularity magnitude)
    NORMALIZE_COOC = False
    # Number of candidates to retrieve via Co-occurrence
    TOP_K_COOC = 60

    # Source B: Repurchase History
    # Number of candidates to retrieve from user's own purchase history (Habitual)
    TOP_K_REPURCHASE = 20

    # Source C: Popularity (Fallback)
    # Number of top popular items to retrieve for coverage
    TOP_K_POPULARITY = 12

    # -------------------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------------------
    # Columns to use for User-Item Affinity Features (Interaction Features)
    # The ranker will compute the match between user profile and item attribute for these:
    AFFINITY_COLS = [
        "department_no",
        "colour_group_code",
        "garment_group_no",
        "section_no",
        "graphical_appearance_no",
        "index_group_no",
    ]

    # -------------------------------------------------------------------------
    # Ranking Model (LightGBM) Hyperparameters
    # -------------------------------------------------------------------------
    LGBM_PARAMS = {
        "objective": "binary",  # Treating ranking as binary classification of candidates
        "metric": "auc",  # Area Under Curve as proxy for ranking quality
        "boosting_type": "gbdt",
        "n_estimators": 2000,  # High number, controlled by early stopping
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,  # Silent execution
        "random_state": SEED,
        "n_jobs": 12,  # Utilize available vCPUs
    }

    # Training settings
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 50

    # -------------------------------------------------------------------------
    # Cache File Paths
    # -------------------------------------------------------------------------
    # These paths are used to store intermediate processed data

    # Mappings
    PATH_ARTICLE_MAP = WORKING_DIR / "article_map.parquet"
    PATH_CUSTOMER_MAP = WORKING_DIR / "customer_map.parquet"

    # Matrices
    PATH_COOC_MATRIX = WORKING_DIR / "cooccurrence_matrix.npz"
    PATH_GLOBAL_POPULARITY = WORKING_DIR / "global_popularity.npy"

    # Candidates
    PATH_CANDIDATES_TRAIN = WORKING_DIR / "candidates_train.parquet"
    PATH_CANDIDATES_TEST = WORKING_DIR / "candidates_test.parquet"

    # Final Submission
    PATH_SUBMISSION = SUBMISSION_DIR / "submission.csv"
