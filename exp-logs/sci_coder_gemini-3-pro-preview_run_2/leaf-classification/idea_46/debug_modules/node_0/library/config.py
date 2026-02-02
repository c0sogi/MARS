import os
import numpy as np

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Specific cache directory for this idea (Idea 46)
CACHE_DIR = os.path.join(WORKING_DIR, "idea_46")

# Ensure necessary writeable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data File Paths
IMAGES_DIR = os.path.join(INPUT_DIR, "images")
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
RANDOM_STATE = 42
N_JOBS = 12
FLOAT_PRECISION = np.float64  # Strictly use float64 as per "Precision" requirement

# =============================================================================
# FEATURE ENGINEERING HYPERPARAMETERS
# =============================================================================
# For View D: Compressed-Interaction Global Experts
PCA_COMPONENTS_INTERACTION = 25
POLY_DEGREE = 2

# For View C: Polynomial Physical Experts
# Threshold to detect if background is white (leaf is black) vs inverted
POLARITY_THRESHOLD = 0.5

# =============================================================================
# MODEL HYPERPARAMETERS (LDA EXPERTS)
# =============================================================================
# The broad library of shrinkage parameters for Views B and D
SHRINKAGE_GRID = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, "auto"]

# Specific fixed shrinkage values for View A (Baseline)
SHRINKAGE_FIXED_BASELINE = [0.001, 0.01]

# Configuration specifications for the Ensemble Views
# This dictionary guides the pipeline construction in the model module
VIEW_SPECS = {
    "A": {
        "name": "Marginal_Anchors",
        "description": "Global Features -> Yeo-Johnson -> LDA(OAS/Fixed)",
        "feature_type": "global",
        "pipeline_steps": ["power_transform"],
        "lda_solver": "lsqr",
        "shrinkage_options": SHRINKAGE_FIXED_BASELINE,
        "covariance_estimator": "oas",
    },
    "B": {
        "name": "Rotational_Experts",
        "description": "Global Features -> Power -> PCA(No Whiten) -> Power -> LDA(Library)",
        "feature_type": "global",
        "pipeline_steps": ["power_transform", "pca_no_whiten", "power_transform"],
        "lda_solver": "lsqr",
        "shrinkage_options": SHRINKAGE_GRID,
    },
    "C": {
        "name": "Polynomial_Physical",
        "description": "Morphometrics -> Power -> Poly(2) -> Power -> LDA(Ledoit-Wolf)",
        "feature_type": "morphometric",
        "pipeline_steps": ["power_transform", "poly_features", "power_transform"],
        "lda_solver": "eigen",  # Eigen solver supports automatic shrinkage like Ledoit-Wolf
        "shrinkage_options": ["auto"],  # Ledoit-Wolf is automatic
        "poly_degree": POLY_DEGREE,
    },
    "D": {
        "name": "Compressed_Interaction",
        "description": "Global -> Power -> PCA(25) -> Poly(2) -> Power -> LDA(Library)",
        "feature_type": "global",
        "pipeline_steps": [
            "power_transform",
            "pca_compress",
            "poly_features",
            "power_transform",
        ],
        "lda_solver": "lsqr",
        "shrinkage_options": SHRINKAGE_GRID,
        "pca_components": PCA_COMPONENTS_INTERACTION,
        "poly_degree": POLY_DEGREE,
    },
}

# =============================================================================
# ENSEMBLE SELECTION SETTINGS
# =============================================================================
SELECTION_ITERATIONS = 50  # Number of steps for Greedy Forward Selection
