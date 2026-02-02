import os
import numpy as np


class Config:
    """
    Configuration for the Hybrid Kernel-Linear Ensemble Leaf Classification Task.
    Defines paths, constants, and hyperparameter grids for LR, LDA, and SVM models.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    RANDOM_SEED = 42

    # ==========================================
    # File Paths
    # ==========================================
    # Input Metadata (Read-only)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories (Read/Write)
    WORKING_DIR = "./working/idea_2"
    CACHE_DIR = WORKING_DIR
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Configuration
    # ==========================================
    # Feature groups to extract/use from the dataset
    FEATURE_GROUPS = ["margin", "shape", "texture"]

    # Scaling Method
    SCALER_TYPE = "standard"  # Uses StandardScaler

    # ==========================================
    # Model Hyperparameters
    # ==========================================

    # 1. Logistic Regression (Discriminative Linear)
    # Strategy: Weak regularization (High C), L2 penalty, LBFGS solver.
    # LogisticRegressionCV will automatically select the best C from 'Cs'.
    LR_PARAMS = {
        "Cs": np.logspace(-2, 4, 20),  # Grid from 0.01 to 10000
        "cv": 3,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 5000,
        "multi_class": "multinomial",
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "scoring": "neg_log_loss",
    }

    # 2. Linear Discriminant Analysis (Generative Linear)
    # Strategy: Automatic covariance shrinkage (Ledoit-Wolf) using lsqr solver.
    LDA_PARAMS = {"solver": "lsqr", "shrinkage": "auto"}

    # 3. Support Vector Machine (Discriminative Non-Linear)
    # Strategy: RBF Kernel, tuned C/Gamma, wrapped in CalibratedClassifierCV.

    # Base SVC Parameters
    SVM_BASE_PARAMS = {
        "kernel": "rbf",
        "class_weight": "balanced",
        "probability": False,  # Probability estimation handled by wrapper
        "cache_size": 1000,
        "random_state": RANDOM_SEED,
    }

    # Grid for Hyperparameter Tuning
    SVM_PARAM_GRID = {
        "C": [1, 10, 100, 1000, 5000],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
    }

    # Inner CV for SVM Grid Search
    SVM_GRID_CV = 3

    # Calibration Wrapper Parameters
    SVM_CALIBRATION_PARAMS = {
        "method": "sigmoid",  # Platt Scaling
        "cv": 3,
        "n_jobs": -1,
    }

    # ==========================================
    # Ensemble Configuration
    # ==========================================
    # Weights for Soft Voting (Averaging)
    ENSEMBLE_WEIGHTS = {"lr": 1.0, "lda": 1.0, "svm": 1.0}

    # Metric Clipping to avoid log(0)
    CLIP_EPSILON = 1e-15

    @staticmethod
    def setup_directories():
        """Ensures necessary working directories exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on module load
Config.setup_directories()
