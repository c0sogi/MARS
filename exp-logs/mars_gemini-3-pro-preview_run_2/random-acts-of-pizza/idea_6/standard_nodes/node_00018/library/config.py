import os

# =============================================================================
# Directories and Paths
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/optimization_1"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Raw Data Paths
TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

# Metadata Paths (CSV files mapping request_ids to labels and split indices)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# =============================================================================
# Global Constants
# =============================================================================
SEED = 42
N_JOBS = 12  # Number of vCPUs available for parallel processing

# =============================================================================
# Feature Engineering Configuration
# =============================================================================
# Transformer Model: Using MPNet for high-quality semantic representations (768 dims)
TRANSFORMER_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Embedding Processing: L2 Normalization ensures embeddings lie on the hypersphere
NORMALIZE_EMBEDDINGS = True

# Tabular Processing: RankGauss (QuantileTransformer with normal output)
# This handles outliers and aligns tabular feature distribution with L2-normalized embeddings
TABULAR_SCALER = "quantile_normal"

# =============================================================================
# Model Hyperparameters
# =============================================================================
# 1. Base Learner: Logistic Regression
# 'liblinear' is efficient for high-dimensional data; 'balanced' handles class imbalance.
LR_BASE_PARAMS = {
    "penalty": "l2",
    "class_weight": "balanced",
    "solver": "liblinear",
    "max_iter": 1000,
    "random_state": SEED,
}

# Regularization Strength Candidates (C) for Grid Search
# We focus on smaller C values (stronger regularization) to prevent overfitting
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

# 2. Ensemble: Bagging Classifier
# Combines bootstrap sampling (rows) and random subspace sampling (features)
BAGGING_PARAMS = {
    "n_estimators": 200,  # Increased to reduce variance further
    "max_samples": 0.8,  # Fraction of samples to draw for each base estimator
    "max_features": 0.8,  # Fraction of features to draw for each base estimator
    "bootstrap": True,  # Sample with replacement
    "bootstrap_features": False,
    "n_jobs": N_JOBS,
    "random_state": SEED,
}

# =============================================================================
# Caching Configuration
# =============================================================================
# File paths for caching processed features to speed up experiments
CACHE_PATHS = {
    "train_features": os.path.join(WORKING_DIR, "train_features.parquet"),
    "val_features": os.path.join(WORKING_DIR, "val_features.parquet"),
    "test_features": os.path.join(WORKING_DIR, "test_features.parquet"),
}
