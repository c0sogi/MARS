import os
import numpy as np

# =============================================================================
# GLOBAL PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Working directory for this specific experimental iteration (Idea 48)
# This is used for caching deterministic data processing steps.
WORKING_DIR = "./working/idea_48"
os.makedirs(WORKING_DIR, exist_ok=True)

# Directory for final submission
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# HYPERPARAMETERS & SETTINGS
# =============================================================================
SEED = 42
FLOAT_PRECISION = np.float64  # Enforce double precision for linear algebra stability

# =============================================================================
# DATA COLUMN DEFINITIONS
# =============================================================================
ID_COL = "id"
TARGET_COL = "species"
FILE_PATH_COL = "file_path"

# 1. Pre-extracted Tabular Features (192 columns)
MARGIN_COLS = [f"margin{i}" for i in range(1, 65)]
SHAPE_COLS = [f"shape{i}" for i in range(1, 65)]
TEXTURE_COLS = [f"texture{i}" for i in range(1, 65)]

TABULAR_FEATURES = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS

# 2. Geometric Primitives (Absolute Scalars)
# Extracted from binary images to preserve biological size signals
GEOMETRIC_PRIMITIVES = [
    "Area",
    "Perimeter",
    "Convex_Area",
    "Convex_Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Equivalent_Diameter",
]

# 3. Ratio Projections (Dimensionless Shape Descriptors)
# Explicitly computed to expose non-linear relationships to the linear solver
# Selection based on Boundedness (Cite solution_lesson_node_00142) and
# Macro-Geometric Completeness (Cite solution_lesson_node_00120).
RATIO_FEATURES = [
    "Solidity",  # Area / Convex_Area
    "Extent",  # Area / (Bounding_Width * Bounding_Height)
    "Aspect_Ratio",  # Bounding_Width / Bounding_Height
    "Roundness",  # 4 * pi * Area / Perimeter^2 (Scale Invariant)
    "Eccentricity",  # sqrt(1 - (Minor / Major)^2) (Bounded [0, 1])
]

# Combined Geometric Features
GEOMETRIC_FEATURES = GEOMETRIC_PRIMITIVES + RATIO_FEATURES

# Total Feature Set for Modeling
# Note: Deterministic ordering is crucial for reproducibility
ALL_FEATURES = sorted(TABULAR_FEATURES + GEOMETRIC_FEATURES)
