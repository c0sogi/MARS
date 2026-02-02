import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the Seismic Eruption Prediction pipeline.
    Implements the 'Latent-Source Cepstral Stacking' strategy settings.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    N_CORES = 12
    EXPERIMENT_ID = "idea_6"

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", EXPERIMENT_ID)
    SUBMISSION_DIR = "./submission"

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Constants
    # ==========================================
    NUM_SENSORS = 10
    SENSOR_COLS = [f"sensor_{i}" for i in range(1, NUM_SENSORS + 1)]

    # 10 minutes of data at ~100Hz = 60001 samples
    SEGMENT_LENGTH = 60001
    SAMPLING_RATE = 100

    # ==========================================
    # Feature Extraction Configuration
    # ==========================================
    # PCA / Latent Source
    N_PCA_COMPONENTS = 1  # We extract PC1 as the "Virtual Source"

    # Audio / Spectral Features
    # Nyquist is 50Hz, so fmax is set to 50
    N_MFCC = 40
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 1
    FMAX = 50

    # ==========================================
    # Model Hyperparameters
    # ==========================================

    # Branch A: LightGBM (Tabular)
    LGBM_PARAMS = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 5000,
        "early_stopping_rounds": 100,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "n_jobs": -1,
        "seed": SEED,
    }

    # Branch B: EfficientNet (Vision)
    NN_PARAMS = {
        "model_name": "efficientnet_b0",
        "in_channels": 10,  # 10 sensors stacked as channels
        "num_classes": 1,  # Regression output
        "batch_size": 32,
        "epochs": 25,  # Extended training for convergence
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "scheduler_T_max": 25,  # For CosineAnnealingLR
        "target_log_scale": True,  # Apply np.log1p to target, np.expm1 to pred
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 4,
    }

    # Meta-Learner (Stacking)
    META_PARAMS = {"alpha": 1.0, "random_state": SEED}  # Ridge regularization strength

    def __init__(self):
        """
        Initialize configuration and ensure necessary directories exist.
        """
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        # Set the random seed upon initialization
        self.set_seed(self.SEED)

    @staticmethod
    def set_seed(seed=42):
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
