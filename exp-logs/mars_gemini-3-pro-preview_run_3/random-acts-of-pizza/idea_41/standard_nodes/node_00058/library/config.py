import os
import numpy as np


class Config:
    # =========================================================================
    # 1. File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate files (idea_41 specific)
    WORKING_DIR = "./working/idea_41"

    # Submission directory
    SUBMISSION_DIR = "./demo_submission"  # Using demo_submission as per file structure or generic submission
    if not os.path.exists(SUBMISSION_DIR):
        SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # 2. Global Constants & Seeds
    # =========================================================================
    RANDOM_STATE = 42
    N_JOBS = 12  # Utilizing available vCPUs

    # =========================================================================
    # 3. Data Column Definitions
    # =========================================================================
    ID_COL = "request_id"
    TARGET_COL = "requester_received_pizza"

    # Text Columns for Concatenation
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Subreddit History Column (list of strings)
    SUBREDDIT_COL = "requester_subreddits_at_request"

    # Numerical / Metadata Allow-List
    # Explicitly selected based on "Augmented Global Metadata" strategy
    # Excludes all *_at_retrieval columns to prevent leakage
    NUMERICAL_COLS = [
        "unix_timestamp_of_request_utc",
        "requester_account_age_in_days_at_request",
        "requester_days_since_first_post_on_raop_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_subreddits_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
    ]

    # =========================================================================
    # 4. Feature Engineering Configuration
    # =========================================================================

    # --- Latent User Clustering (Subreddit History) ---
    # TF-IDF settings for subreddit lists
    SUBREDDIT_TFIDF_PARAMS = {
        "max_features": 1000,
        "binary": True,  # Presence/Absence is more important than count
        "stop_words": None,
        "token_pattern": r"(?u)\b\w\w+\b",
    }

    # Dimensionality Reduction for Clustering
    SVD_COMPONENTS = 20

    # Clustering Settings
    N_CLUSTERS = 10  # Number of latent personas

    # --- Text Feature Engineering ---
    # Sparse Text Bagger (TF-IDF)
    TEXT_TFIDF_PARAMS = {
        "sublinear_tf": True,
        "min_df": 5,
        "max_features": 5000,  # Limit to prevent explosion
        "stop_words": "english",
        "ngram_range": (1, 2),
    }

    # Dense Embeddings
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    BATCH_SIZE = 32

    # =========================================================================
    # 5. Model Hyperparameters (Level 1 Base Learners)
    # =========================================================================

    # Common Training Params
    N_FOLDS = 5
    EARLY_STOPPING_ROUNDS = 50

    # 1. Sparse Lexical Branch: Lexical Bagger (Random Forest)
    RF_LEXICAL_PARAMS = {
        "n_estimators": 300,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "verbose": 0,
    }

    # 2. Sparse Behavioral Branch: Community Bagger (Random Forest)
    RF_COMMUNITY_PARAMS = {
        "n_estimators": 300,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "verbose": 0,
    }

    # 3. Dense Semantic Branch: Semantic Booster (XGBoost)
    XGB_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 3.0,  # Approx ratio of neg/pos (0.75/0.25)
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "tree_method": "hist",
        "device": "cuda",  # Use GPU
    }

    # 4. Dense Semantic Branch: Semantic Gradient (LightGBM)
    LGBM_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 3.0,
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "verbose": -1,
        "device": "gpu",  # Use GPU
    }

    # 5. Dense Semantic Branch: Semantic Bagger (Random Forest)
    # Regularized to prevent memorizing dense noise
    RF_SEMANTIC_PARAMS = {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": N_JOBS,
        "verbose": 0,
    }

    # 6. Contextual Branch: Metadata Anchor (Logistic Regression)
    LR_PARAMS = {
        "penalty": "l2",
        "C": 0.1,
        "class_weight": "balanced",
        "solver": "liblinear",
        "random_state": RANDOM_STATE,
    }

    # =========================================================================
    # 6. Meta-Learner Configuration (Level 2)
    # =========================================================================
    META_LEARNER_PARAMS = {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "random_state": RANDOM_STATE,
    }

    # =========================================================================
    # 7. Utilities
    # =========================================================================
    @staticmethod
    def get_cache_path(filename):
        """Returns full path for a file in the working/cache directory."""
        return os.path.join(Config.WORKING_DIR, filename)
