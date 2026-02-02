import os
import numpy as np

# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Specific directory for this idea's caching
IDEA_NAME = "idea_52"
CACHE_DIR = os.path.join(WORKING_DIR, IDEA_NAME)

# Ensure output directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# 2. GLOBAL SETTINGS & HYPERPARAMETERS
# =============================================================================
RANDOM_SEED = 42
FLOAT_PRECISION = np.float64
PROB_CLIP_EPS = 1e-15  # For log loss stability

# =============================================================================
# 3. FEATURE DEFINITIONS
# =============================================================================
# The dataset contains 3 sets of 64 features each
MARGIN_COLS = [f"margin_{i}" for i in range(1, 65)]
SHAPE_COLS = [f"shape_{i}" for i in range(1, 65)]
TEXTURE_COLS = [f"texture_{i}" for i in range(1, 65)]

# Combined global view
GLOBAL_FEATURE_COLS = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS

# =============================================================================
# 4. PREPROCESSING HYPERPARAMETERS
# =============================================================================
# Image Processing
POLARITY_CHECK_THRESHOLD = 0.5  # If corner mean > 0.5, invert image

# Dimensionality Reduction & Expansion
PCA_VARIANCE_THRESHOLD = 0.95  # Retain 95% variance in PCA steps
POLY_DEGREE = 2  # Degree for PolynomialFeatures
LDA_SUBSPACE_COMPONENTS = 15  # n_components for the Discriminative Subspace expert

# =============================================================================
# 5. MODEL CONFIGURATION (LDA)
# =============================================================================
# Shrinkage options for LDA solvers (OAS/Ledoit-Wolf/Fixed)
# 'auto' implies automatic estimation (Ledoit-Wolf or OAS)
# Floats imply fixed shrinkage
LDA_SHRINKAGE_GRID = [1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.5, "auto"]

# =============================================================================
# 6. ENSEMBLE EXPERT LIBRARY CONFIGURATION
# =============================================================================
# Defines the search space for the Greedy Forward Selection
# Each entry represents a group of experts to be trained and evaluated

EXPERT_LIBRARY_CONFIG = [
    # --- Group A: Global Linear Anchors ---
    # Baseline linear models on all 192 features
    {
        "group": "A",
        "name": "Global_Marginal_Linear",
        "feature_source": "global",  # Uses GLOBAL_FEATURE_COLS
        "pipeline_type": "marginal_linear",  # PowerTransformer -> LDA
        "shrinkage_grid": LDA_SHRINKAGE_GRID,
    },
    {
        "group": "A",
        "name": "Global_Rotational_Linear",
        "feature_source": "global",
        "pipeline_type": "rotational_linear",  # PT -> PCA(whiten=False) -> PT -> LDA
        "shrinkage_grid": LDA_SHRINKAGE_GRID,
    },
    # --- Group B: Physical Polynomial Experts ---
    # Non-linear physical constraints from raw images
    {
        "group": "B",
        "name": "Morphometric_Poly",
        "feature_source": "morphometrics",  # Extracted from images
        "pipeline_type": "physical_poly",  # PT -> Poly(2) -> PT -> LDA
        "shrinkage_grid": [
            "auto"
        ],  # Physical features often benefit from auto shrinkage
    },
    # --- Group C: Component-Wise Polynomial Experts ---
    # Intra-component non-linear interactions
    {
        "group": "C",
        "name": "Margin_Poly",
        "feature_source": "margin",  # Uses MARGIN_COLS
        "pipeline_type": "component_poly",  # PT -> PCA(0.95) -> Poly(2) -> PT -> LDA
        "shrinkage_grid": LDA_SHRINKAGE_GRID,
    },
    {
        "group": "C",
        "name": "Shape_Poly",
        "feature_source": "shape",  # Uses SHAPE_COLS
        "pipeline_type": "component_poly",
        "shrinkage_grid": LDA_SHRINKAGE_GRID,
    },
    {
        "group": "C",
        "name": "Texture_Poly",
        "feature_source": "texture",  # Uses TEXTURE_COLS
        "pipeline_type": "component_poly",
        "shrinkage_grid": LDA_SHRINKAGE_GRID,
    },
    # --- Group D: Discriminative-Subspace Expert ---
    # Global non-linear interactions in discriminative subspace
    {
        "group": "D",
        "name": "Global_Subspace_Poly",
        "feature_source": "global",
        "pipeline_type": "subspace_poly",  # PT -> LDA(15) -> Poly(2) -> PT -> LDA
        "shrinkage_grid": LDA_SHRINKAGE_GRID,
    },
]
