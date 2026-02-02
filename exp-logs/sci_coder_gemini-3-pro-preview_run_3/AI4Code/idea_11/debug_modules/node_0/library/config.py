import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in CuDNN if needed, though often slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Config:
    """
    Global configuration for the Multi-Scale Structural Heatmap Regressor (MS-SHR) pipeline.
    """

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on minimal subsets for debugging

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"
    SUBMISSION_DIR = "./submission"

    # Input Metadata
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "fine_tuned_mpnet")
    LGBM_MODEL_PATH = os.path.join(WORKING_DIR, "lgbm_model.txt")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (for deterministic data processing)
    TRAIN_PAIRS_PATH = os.path.join(WORKING_DIR, "train_pairs.parquet")
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # -------------------------------------------------------------------------
    # Model Hyperparameters (Backbone)
    # -------------------------------------------------------------------------
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    MAX_LENGTH = 128
    BATCH_SIZE = 32

    # -------------------------------------------------------------------------
    # Training Hyperparameters (Fine-Tuning)
    # -------------------------------------------------------------------------
    NUM_EPOCHS = 1
    LEARNING_RATE = 2e-5
    FT_SAMPLE_SIZE = 40000  # Number of notebooks to use for contrastive fine-tuning

    # -------------------------------------------------------------------------
    # Feature Engineering Hyperparameters
    # -------------------------------------------------------------------------
    NUM_BINS = 20  # Resolution of the structural heatmap
    SMOOTHING_SCALES = [1, 3, 5]  # Kernel sizes for multi-scale convolution

    # -------------------------------------------------------------------------
    # LightGBM Hyperparameters
    # -------------------------------------------------------------------------
    LGBM_PARAMS = {
        "objective": "regression",
        "metric": "mse",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "n_jobs": 12,  # Using available vCPUs
        "random_state": SEED,
    }
    # Note: early_stopping_rounds is typically passed to the .train() or .fit() method,
    # not the constructor, but we define the value here for reference.
    EARLY_STOPPING_ROUNDS = 50

    @classmethod
    def setup(cls):
        """
        Ensures that necessary working and output directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_lgbm_params(cls):
        """
        Returns a copy of LGBM parameters to avoid mutation issues.
        """
        return cls.LGBM_PARAMS.copy()
