import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Create directories if they don't exist
    @classmethod
    def setup(cls):
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    # Input Files
    RAW_TRAIN_PATH = os.path.join(INPUT_DIR, "train.json")
    RAW_TEST_PATH = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cached Feature Files
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # ==========================================
    # Feature Engineering Parameters
    # ==========================================
    # Text Embedding
    TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
    TEXT_COLS = ["request_title", "request_text_edit_aware"]

    # Dimensionality Reduction for Branch B
    N_PCA_COMPONENTS = 50

    # Numerical Features to be Scaled
    NUMERICAL_COLS = [
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

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Stacking Setup
    N_FOLDS = 5

    # Branch A: Logistic Regression (High-Dimensional)
    # High bias, low variance baseline
    LR_PARAMS = {
        "C": 1.0,
        "solver": "liblinear",
        "penalty": "l2",
        "max_iter": 1000,
        "random_state": SEED,
        "class_weight": "balanced",
    }

    # Branch B: LightGBM (Low-Dimensional / PCA)
    # Non-linear interactions on compressed space
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 3,  # Strict regularization to prevent overfitting
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 1000,  # Will use early stopping
        "n_jobs": -1,
        "random_state": SEED,
        "is_unbalance": True,
    }

    # LightGBM Training Params
    LGBM_FIT_PARAMS = {
        "callbacks": None,  # To be populated with early_stopping callback in training script
    }
    LGBM_EARLY_STOPPING_ROUNDS = 100

    # Meta-Learner: Logistic Regression
    # Combines probas from A and B
    META_LR_PARAMS = {
        "C": 1.0,
        "solver": "lbfgs",
        "penalty": "l2",
        "random_state": SEED,
    }
