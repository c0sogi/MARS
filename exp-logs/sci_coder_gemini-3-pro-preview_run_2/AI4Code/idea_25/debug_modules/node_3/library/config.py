import os

# ------------------------------------------------------------------------------
# Directory Configuration
# ------------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_25"
SUBMISSION_DIR = "./submission"

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Global Reproducibility
# ------------------------------------------------------------------------------
RANDOM_STATE = 42

# ------------------------------------------------------------------------------
# Data Configuration
# ------------------------------------------------------------------------------
VAL_SIZE = 0.2

# ------------------------------------------------------------------------------
# Stage 1: Vectorization (TF-IDF) Configuration
# ------------------------------------------------------------------------------
# Configuration based on lessons:
# - Vocabulary=60,000
# - N-gram range=(1, 2)
# - Sublinear TF=True
# - No Accent Stripping (strip_accents=None)
TFIDF_PARAMS = {
    "min_df": 2,
    "max_df": 0.9,
    "max_features": 60000,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "strip_accents": None,
    "use_idf": True,
    "smooth_idf": True,
    "token_pattern": r"(?u)\b\w\w+\b",
}

# ------------------------------------------------------------------------------
# Stage 1: Latent Semantic Analysis (SVD) Configuration
# ------------------------------------------------------------------------------
# Decoupled latent view for recall
SVD_N_COMPONENTS = 128
SVD_RANDOM_STATE = RANDOM_STATE

# ------------------------------------------------------------------------------
# Feature Extraction Configuration (Multi-Resolution Neighborhoods)
# ------------------------------------------------------------------------------
# Number of neighbors for smoothing (Mean/Std features)
NUM_NEIGHBORS = 10

# Number of specific top instances to keep as explicit features
# (Rank and Similarity of 1st and 2nd neighbors)
TOP_K_INSTANCES = 2

# ------------------------------------------------------------------------------
# Model Hyperparameters
# ------------------------------------------------------------------------------

# Stage 1 Model: Ridge Regression (The "Signpost" Model)
RIDGE_ALPHA = 1.0

# Stage 2 Model: LightGBM Regressor (The "Refinement" Model)
# Objective: Minimize Mean Absolute Error (MAE) on Normalized Rank
LGBM_PARAMS = {
    "objective": "regression_l1",  # L1 loss corresponds to MAE
    "metric": "mae",
    "boosting_type": "gbdt",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

# Training Loop Configuration
EARLY_STOPPING_ROUNDS = 50
VERBOSE_EVAL = 100
