import os
import numpy as np

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================
BASE_DIR = os.path.abspath(".")
INPUT_DIR = os.path.join(BASE_DIR, "input")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
WORKING_DIR = os.path.join(BASE_DIR, "working", "idea_8")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data Paths
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_OUTPUT_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# DATA PROCESSING CONFIGURATION
# =============================================================================
# PCA Configuration for the GPC branch
# Retain 95% variance to densify feature space and remove noise
PCA_VARIANCE = 0.95

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# 1. Logistic Regression (Discriminative Linear Component)
# Uses LogisticRegressionCV to optimize for neg_log_loss
LOGREG_CONFIG = {
    "Cs": np.logspace(-2, 4, 20),  # Constrained to high-signal regime
    "cv": 3,
    "scoring": "neg_log_loss",  # Explicitly align with competition metric
    "solver": "lbfgs",
    "penalty": "l2",
    "max_iter": 10000,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "refit": True,
}

# 2. Linear Discriminant Analysis (Generative Linear Component)
# Uses Ledoit-Wolf shrinkage for high-dimensional small-sample data
LDA_CONFIG = {
    "solver": "lsqr",  # 'lsqr' supports shrinkage
    "shrinkage": "auto",  # 'auto' results in Ledoit-Wolf shrinkage
}

# 3. Gaussian Process Classifier (Probabilistic Non-Linear Component)
# Uses RBF kernel (instantiated in model code) and internal Bayesian optimization
GPC_CONFIG = {
    "n_restarts_optimizer": 1,
    "max_iter_predict": 100,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "copy_X_train": False,
}
