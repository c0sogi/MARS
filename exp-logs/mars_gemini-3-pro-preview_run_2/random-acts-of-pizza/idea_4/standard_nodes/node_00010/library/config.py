import os
import random
import numpy as np
import torch


class Config:
    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Raw Data Inputs
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Inputs
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache File Paths (for deterministic data processing)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    N_SPLITS = 5  # Stratified K-Fold
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Feature Extraction
    # ==========================================
    TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    # ==========================================
    # Model Hyperparameters (Defaults)
    # ==========================================

    # Branch A: Linear Anchor (Logistic Regression)
    LOGREG_PARAMS = {
        "penalty": "l2",
        "solver": "liblinear",
        "max_iter": 2000,
        "random_state": SEED,
        "verbose": 0,
    }

    # Branch B: Non-Linear Expert (PLS + SVM)
    PLS_N_COMPONENTS = 10

    SVM_PARAMS = {
        "kernel": "rbf",
        "probability": True,  # Required for soft voting ensemble
        "random_state": SEED,
        "class_weight": "balanced",
        "verbose": False,
    }

    # Ensemble Configuration
    WEIGHT_LINEAR = 0.6
    WEIGHT_SVM = 0.4

    # ==========================================
    # Hyperparameter Tuning Grids
    # ==========================================
    LOGREG_GRID = {"C": [0.01, 0.1, 1.0, 10.0, 100.0]}

    SVM_GRID = {"C": [0.1, 1.0, 10.0, 50.0], "gamma": ["scale", "auto", 0.01, 0.1]}

    PLS_GRID = {"n_components": [5, 10, 15, 20]}

    @classmethod
    def setup(cls):
        """Ensures that working and submission directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


def set_seed(seed=Config.SEED):
    """Sets random seeds for reproducibility across Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
