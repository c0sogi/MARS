import os
import random
import numpy as np
import torch

# =============================================================================
# 1. PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_24")
SUBMISSION_DIR = "./submission"

# Ensure necessary writeable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================
SEED = 42


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 3. DATA PROCESSING HYPERPARAMETERS
# =============================================================================
# TF-IDF Vectorization (The "Lexical View")
# High vocab and bigrams for precision "signposting"
VOCAB_SIZE = 60000
NGRAM_RANGE = (1, 2)
SUBLINEAR_TF = True
USE_IDF = True
STRIP_ACCENTS = None  # Preserved from lessons: Do not strip accents

# Latent Semantic Analysis (The "Latent View")
# Scalable dense representation for conceptual alignment
SVD_COMPONENTS = 128
SVD_RANDOM_STATE = SEED

# Multi-Resolution Neighborhood Features
# We extract features from both Lexical and Latent similarity matrices
NUM_NEIGHBORS = 10  # Number of neighbors to retrieve (Top-K)
NEIGHBOR_RANKS_TO_KEEP = [
    0,
    1,
    2,
]  # Specific indices to keep as instance features (0=1st NN)
SMOOTHING_K = 10  # Number of neighbors to use for Mean/Std smoothing

# =============================================================================
# 4. MODEL HYPERPARAMETERS
# =============================================================================
# Stage 1: Sparse Lexical Regressor (Ridge)
# Used to generate OOF predictions as a robust prior
RIDGE_ALPHA = 1.0
NUM_FOLDS = 5  # 5-Fold CV for OOF generation

# Stage 2: Gated Multi-Resolution Gradient Booster (LightGBM)
# Refines predictions using neighborhood signals and metadata
LGBM_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "mae",  # Minimize Mean Absolute Error for ranking
    "metric": "mae",
    "n_jobs": -1,
    "random_state": SEED,
    "verbosity": -1,
    "device": "gpu",  # Leverage NVIDIA A100
    "gpu_use_dp": True,  # Double precision for GPU
}

# Training Control
VAL_SIZE = 0.2  # Validation split size (grouped by ancestor)
EARLY_STOPPING_ROUNDS = 100  # Stop if validation MAE doesn't improve
VERBOSE_EVAL = 100  # Print metrics every 100 rounds

# =============================================================================
# 5. COMPUTE & RESOURCES
# =============================================================================
NUM_WORKERS = 4  # Workers for data loading (12 vCPUs available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
