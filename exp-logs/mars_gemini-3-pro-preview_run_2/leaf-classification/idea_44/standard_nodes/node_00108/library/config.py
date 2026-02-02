import os
import numpy as np
import random
import torch

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Working directory for caching intermediate files (parquet/npy)
WORKING_DIR = "./working/idea_44"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission directory
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
RANDOM_SEED = 42
VAL_SIZE = 0.2
FLOAT_PRECISION = np.float64  # Double precision as per strategy

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

# Preprocessing: Quantile Transformer (Robust Distributional)
QUANTILE_PARAMS = {
    "n_quantiles": 50,
    "output_distribution": "normal",
    "random_state": RANDOM_SEED,
}

# Preprocessing: Power Transformer (Statistical Anchors)
POWER_PARAMS = {"method": "yeo-johnson", "standardize": True}

# Feature Engineering: Polynomial Expansion
POLY_PARAMS = {"degree": 2, "interaction_only": False, "include_bias": False}

# =============================================================================
# EXPERT LIBRARY CONFIGURATION
# =============================================================================
# Defines the candidates for the Dynamic Ensemble Selection
# keys:
#   - feature_view: 'global' (provided 192 feats) or 'morph_poly' (extracted + poly)
#   - preprocessing: 'power' or 'quantile'
#   - model_type: 'lda_fixed', 'lda_lw' (Ledoit-Wolf), 'lda_oas' (OAS)
#   - shrinkage: float (for fixed) or None (for auto/estimators)

EXPERT_LIBRARY = [
    # Group A: Statistical Anchors (Global + Power + LDA Fixed Shrinkage)
    {
        "id": "stat_anchor_001",
        "feature_view": "global",
        "preprocessing": "power",
        "model_type": "lda_fixed",
        "shrinkage": 0.001,
    },
    {
        "id": "stat_anchor_01",
        "feature_view": "global",
        "preprocessing": "power",
        "model_type": "lda_fixed",
        "shrinkage": 0.01,
    },
    # Group B: Polynomial-Physical Experts (Morph Poly + Power + LDA Ledoit-Wolf)
    {
        "id": "poly_physical_lw",
        "feature_view": "morph_poly",
        "preprocessing": "power",
        "model_type": "lda_lw",
        "shrinkage": "auto",
    },
    # Group C: Robust Distributional Experts (Global + Quantile + LDA OAS)
    {
        "id": "robust_dist_oas",
        "feature_view": "global",
        "preprocessing": "quantile",
        "model_type": "lda_oas",
        "shrinkage": None,
    },
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def set_seed(seed=RANDOM_SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and Torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cache_path(filename):
    """
    Returns the full path for a cached file within the working directory.
    """
    return os.path.join(WORKING_DIR, filename)
