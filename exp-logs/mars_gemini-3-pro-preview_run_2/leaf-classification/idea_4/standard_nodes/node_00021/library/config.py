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
# Optimized based on Lesson 00010 and 00008 (removing balanced weights)
LR_PARAMS = {
    "cv": 3,
    "Cs": 20,  # Denser grid (Cite solution_lesson_node_00010)
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 5000,
    "scoring": "neg_log_loss",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# 2. Linear Discriminant Analysis (Generative Linear)
# Uses Ledoit-Wolf shrinkage (Cite solution_lesson_node_00006)
LDA_PARAMS = {"solver": "lsqr", "shrinkage": "auto"}
