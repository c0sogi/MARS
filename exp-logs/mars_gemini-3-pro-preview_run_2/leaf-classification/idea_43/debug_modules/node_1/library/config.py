import os

# ==============================================================================
# DIRECTORY AND FILE PATHS
# ==============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Working directory for caching intermediate files (parquet/npy)
# Idea ID is 43 based on the prompt context
WORKING_DIR = "./working/idea_43"
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata file paths
TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
VAL_FILE = os.path.join(METADATA_DIR, "val.csv")
TEST_FILE = os.path.join(METADATA_DIR, "test.csv")

# Output submission path
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
RANDOM_SEED = 42
N_JOBS = 12  # Utilize available vCPUs
FLOAT_PRECISION = "float64"  # Enforce double precision for numerical stability

# ==============================================================================
# FEATURE ENGINEERING SETTINGS
# ==============================================================================
# Polarity correction ensures the leaf is always the foreground (value 1)
# before calculating morphometric features.
USE_POLARITY_CORRECTION = True

# List of morphological features to extract
MORPHOMETRIC_FEATURES = [
    "area",
    "perimeter",
    "aspect_ratio",
    "eccentricity",
    "extent",
    "solidity",
    "hu_moment_1",
    "hu_moment_2",
    "hu_moment_3",
    "hu_moment_4",
    "hu_moment_5",
    "hu_moment_6",
    "hu_moment_7",
]

# ==============================================================================
# MODEL HYPERPARAMETERS
# ==============================================================================
# LDA Shrinkage Solvers to include in the expert library
# 'auto' corresponds to the OAS (Oracle Approximating Shrinkage) estimator
# Floats correspond to fixed shrinkage parameters (l2 regularization)
LDA_SHRINKAGE_PARAMS = ["auto", 0.001, 0.01, 0.1]

# ==============================================================================
# GAUSSIANIZATION TOPOLOGY SETTINGS
# ==============================================================================
# Topology A: Marginal Parametric Anchors
# Standard Yeo-Johnson transform on features independently
TOPOLOGY_MARGINAL = {"method": "yeo-johnson", "standardize": True}

# Topology B: Rotational Parametric Experts
# Pipeline: PowerTransform -> PCA(whiten=False) -> PowerTransform
# This aligns data to principal axes before the final Gaussianization
TOPOLOGY_ROTATIONAL = {
    "initial_pt_method": "yeo-johnson",
    "pca_whiten": False,  # Crucial: Do not whiten (scale by variance) to avoid noise amplification
    "pca_components": None,  # None = Keep all components for full rotation
    "final_pt_method": "yeo-johnson",
}

# ==============================================================================
# ENSEMBLE SELECTION SETTINGS
# ==============================================================================
# Maximum number of iterations for Greedy Forward Selection
MAX_ENSEMBLE_ITERATIONS = 100

# Early stopping patience for the selection loop (stop if no improvement)
SELECTION_PATIENCE = 10
