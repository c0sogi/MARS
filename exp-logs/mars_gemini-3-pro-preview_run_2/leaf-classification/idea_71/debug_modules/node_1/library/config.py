import os
import numpy as np
import random

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Working directory for caching intermediate processing steps
# Specific cache directory for this idea iteration
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_71")

# Submission directory
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL HYPERPARAMETERS
# =============================================================================
RANDOM_SEED = 42
VAL_SIZE = 0.2  # Fraction of data used for validation/selection
N_JOBS = 12  # Number of parallel jobs (based on available vCPUs)

# =============================================================================
# FR-SPPE MODEL CONFIGURATION
# =============================================================================
# Polynomial Expansion Degree for Groups B and C
POLY_DEGREE = 2

# Shrinkage / Regularization Grids
# -----------------------------------------------------------------------------
# Group A: Global Linear Anchors
# Input: Global View (192 features)
# Algorithm: LDA + OAS
# Rationale: Low shrinkage sufficient for raw feature space.
GRID_GROUP_A = [0.001, 0.01]

# Group B: Physical Polynomial Experts
# Input: Polarity-Corrected Morphometrics (Hu Moments + Geometric Scalars)
# Algorithm: Regularized QDA
# Rationale: Low dimensionality (D~11) allows QDA, regularization handles stability.
GRID_GROUP_B = [0.1, 0.5]

# Group C: Stratified Full-Rank Polynomial Experts
# Input: Margin, Shape, or Texture (independently expanded via Poly degree=2)
# Algorithm: LDA + OAS
# Rationale: High dimensionality (D~2080) requires higher shrinkage/regularization.
GRID_GROUP_C = [0.1, 0.5]

# Ensemble Selection
# -----------------------------------------------------------------------------
# Parameters for Greedy Forward Selection
MAX_SELECTION_STEPS = 200  # Maximum number of experts to add
SELECTION_TOLERANCE = 1e-6  # Minimum Log Loss improvement required to continue


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def set_seed(seed=RANDOM_SEED):
    """
    Sets the random seed for Python, NumPy, and other relevant libraries
    to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Attempt to seed PyTorch if installed
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # Attempt to seed TensorFlow if installed
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def setup_directories():
    """
    Ensures that the necessary working and submission directories exist.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
