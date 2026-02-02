import os
import numpy as np

# =============================================================================
# DIRECTORY PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Working directory for the DSPGE strategy (Idea 51)
# This directory will store cached features, models, and selection results
WORKING_DIR = "./working/idea_51"
os.makedirs(WORKING_DIR, exist_ok=True)

# Cache directory for deterministic data processing
CACHE_DIR = WORKING_DIR

# Output path for the final submission
SUBMISSION_PATH = "./submission/submission.csv"
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
RANDOM_SEED = 42
N_JOBS = 12  # Utilizing available vCPUs
DTYPE = np.float64  # Strict double precision to minimize numerical noise

# Probability clipping to avoid log(0) extremes as per metric definition
PROB_CLIP = 1e-15

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
# Feature groups provided in the tabular dataset
FEATURE_GROUPS = ["margin", "shape", "texture"]
N_FEATURES_PER_GROUP = 64
TOTAL_GLOBAL_FEATURES = 192

# Image Processing Parameters
# Threshold for corner pixel intensity to detect if background is white (needs inversion)
POLARITY_THRESHOLD = 0.5

# =============================================================================
# EXPERT LIBRARY CONFIGURATION (DSPGE)
# =============================================================================

# Grid of shrinkage parameters for Linear Discriminant Analysis
# Includes specific fixed values and 'auto' (Ledoit-Wolf/OAS)
SHRINKAGE_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.5, 1.0, "auto"]

# Topology C: Discriminative-Subspace Projection
# Number of discriminative components to retain before polynomial expansion
DISCRIMINATIVE_COMPONENTS = 15

# Polynomial Expansion Degree for Topologies C and D
POLY_DEGREE = 2

# Topology Definitions
# Defines the preprocessing pipelines for the ensemble experts
TOPOLOGIES = {
    "A": {
        "name": "Marginal_Statistical_Anchors",
        "description": "Global Features -> PowerTransformer -> LDA",
        "use_pca": False,
        "use_poly": False,
        "use_discriminative_projection": False,
        "input_type": "global",
    },
    "B": {
        "name": "Rotational_Statistical_Experts",
        "description": "Global Features -> PowerTransformer -> PCA(no_whiten) -> PowerTransformer -> LDA",
        "use_pca": True,
        "pca_whiten": False,
        "use_poly": False,
        "use_discriminative_projection": False,
        "input_type": "global",
    },
    "C": {
        "name": "Discriminative_Subspace_Experts",
        "description": "Global -> PT -> LDA_Transform(n=15) -> Poly(2) -> PT -> LDA",
        "use_pca": False,
        "use_poly": True,
        "use_discriminative_projection": True,
        "n_discriminative_components": DISCRIMINATIVE_COMPONENTS,
        "poly_degree": POLY_DEGREE,
        "input_type": "global",
    },
    "D": {
        "name": "Polynomial_Physical_Experts",
        "description": "Morphometrics -> PT -> Poly(2) -> PT -> LDA",
        "use_pca": False,
        "use_poly": True,
        "poly_degree": POLY_DEGREE,
        "use_discriminative_projection": False,
        "input_type": "morphometric",
    },
}

# =============================================================================
# TRAINING / SELECTION CONFIGURATION
# =============================================================================
VAL_SIZE = 0.2
SELECTION_METRIC = "log_loss"
