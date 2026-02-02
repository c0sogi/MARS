import os
import random
import numpy as np
import torch


class Config:
    # =========================================================================
    # Global Configuration
    # =========================================================================
    SEED = 42
    N_FOLDS = 5

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Output directories (Write Allowed)
    # Using specific working directory for this idea
    WORKING_DIR = "./working/idea_47"
    SUBMISSION_DIR = "./submission"

    # Specific file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Configuration
    # =========================================================================
    TARGET_COL = "requester_received_pizza"
    ID_COL = "request_id"

    # Text Columns for Concatenation
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Explicit Allow-List for Dense Features
    # Includes standard user stats, raw timestamp, and RESTORED RAOP history
    DENSE_FEATURES = [
        # User Account Stats
        "requester_account_age_in_days_at_request",
        "requester_upvotes_minus_downvotes_at_request",
        "requester_upvotes_plus_downvotes_at_request",
        "requester_number_of_comments_at_request",
        "requester_number_of_posts_at_request",
        "requester_number_of_subreddits_at_request",
        # Raw Timestamp (for Temporal Booster)
        "unix_timestamp_of_request_utc",
        # Restored RAOP History (Critical Signals)
        "requester_number_of_posts_on_raop_at_request",
        "requester_number_of_comments_in_raop_at_request",
        "requester_days_since_first_post_on_raop_at_request",
    ]

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================

    # 1. Sparse Lexical Branch (Random Forest)
    # Regularized to prevent overfitting on high-dim text data
    LEXICAL_RF_PARAMS = {
        "n_estimators": 500,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
    }

    # 2. Sparse Behavioral Branch (Community RF)
    # Operates on Subreddit History (Bag of Concepts)
    COMMUNITY_RF_PARAMS = {
        "n_estimators": 500,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
    }
    # Vocabulary limit for Community TF-IDF
    COMMUNITY_MAX_FEATURES = 1000

    # 3. Dense Semantic Branch (XGBoost Booster)
    # High capacity, gradient boosting on embeddings
    SEMANTIC_XGB_PARAMS = {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 3.0,  # Handling imbalance (~1:3 ratio)
        "tree_method": "hist",
        "device": "cuda",  # Utilizing A100 GPU
        "random_state": SEED,
        "verbosity": 0,
        "early_stopping_rounds": 50,
    }

    # 4. Dense Semantic Branch (Random Forest Bagger)
    # Structural diversity with modality-specific regularization
    SEMANTIC_RF_PARAMS = {
        "n_estimators": 500,
        "max_depth": 12,  # Prevent memorization of dense noise
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": SEED,
        "verbose": 0,
    }

    # 5. Contextual Branch (Metadata Anchor - Logistic Regression)
    # High-bias regularizer
    METADATA_LOGREG_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": SEED,
        "n_jobs": -1,
    }

    # 6. Contextual Branch (Temporal Booster - LightGBM)
    # Captures non-linear drift in timestamps
    TEMPORAL_LGBM_PARAMS = {
        "n_estimators": 1000,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "random_state": SEED,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    # Level 2 Meta-Learner (Logistic Regression)
    META_LEARNER_PARAMS = {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "random_state": SEED,
    }


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
