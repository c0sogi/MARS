import os

# ==============================================================================
# DIRECTORY CONFIGURATION
# ==============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_4")
SUBMISSION_DIR = "./submission"

# Create necessary directories
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==============================================================================
# FILE PATH CONFIGURATION
# ==============================================================================
# Input Metadata (Pre-generated)
TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
TEST_FILE = os.path.join(METADATA_DIR, "test.csv")

# Raw Input
SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Output
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================
RANDOM_SEED = 42
ID_COL = "id"
TARGET_COL = "species"

# Metric constraints
PROB_CLIP_EPS = 1e-15  # For log loss calculation stability

# ==============================================================================
# MODEL HYPERPARAMETERS
# ==============================================================================

# 1. Logistic Regression (Discriminative Linear)
# Uses LogisticRegressionCV for internal cross-validation
LR_PARAMS = {
    "cv": 3,
    "Cs": 10,  # Coarse logarithmic grid
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 10000,
    "scoring": "neg_log_loss",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# 2. Linear Discriminant Analysis (Generative Linear)
# Uses Ledoit-Wolf shrinkage
LDA_PARAMS = {"solver": "lsqr", "shrinkage": "auto"}

# 3. Gaussian Process Classifier (Probabilistic Non-Linear)
# Preprocessing: PCA is applied ONLY to this branch
PCA_EXPLAINED_VARIANCE = 0.95

GPC_PARAMS = {
    "optimizer": "fmin_l_bfgs_b",
    "n_restarts_optimizer": 0,
    "max_iter_predict": 100,
    "random_state": RANDOM_SEED,
    "copy_X_train": False,
    "n_jobs": -1,
}
