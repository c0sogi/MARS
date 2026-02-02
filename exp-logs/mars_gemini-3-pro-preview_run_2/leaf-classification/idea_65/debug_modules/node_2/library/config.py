import os
import numpy as np

# =============================================================================
# 1. DIRECTORY AND FILE PATHS
# =============================================================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_65"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data file paths
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Image directory (relative to INPUT_DIR as per metadata)
IMAGES_REL_DIR = "images"

# =============================================================================
# 2. GLOBAL CONSTANTS & REPRODUCIBILITY
# =============================================================================
RANDOM_SEED = 42
N_JOBS = 12  # Available vCPUs
FLOAT_PRECISION = np.float64  # Strict float64 requirement (Idea 65)
PROB_CLIP_EPS = 1e-15  # For log-loss stability

# =============================================================================
# 3. DATASET CONFIGURATION
# =============================================================================
NUM_CLASSES = 99

# Provided Feature Columns
MARGIN_COLS = [f"margin{i}" for i in range(1, 65)]
SHAPE_COLS = [f"shape{i}" for i in range(1, 65)]
TEXTURE_COLS = [f"texture{i}" for i in range(1, 65)]
ALL_PROVIDED_FEATURES = MARGIN_COLS + SHAPE_COLS + TEXTURE_COLS

# Extracted Morphometric Features (Group B & C component)
# 7 Hu Moments + 4 Geometric Scalars
MORPHO_COLS = [f"hu_{i}" for i in range(1, 8)] + [
    "aspect_ratio",
    "solidity",
    "extent",
    "eccentricity",
]

# Feature Groups Mapping
FEATURE_GROUPS = {
    "margin": MARGIN_COLS,
    "shape": SHAPE_COLS,
    "texture": TEXTURE_COLS,
    "physical": MORPHO_COLS,
}

# =============================================================================
# 4. PREPROCESSING & MODEL HYPERPARAMETERS
# =============================================================================
# Image Preprocessing
POLARITY_CHECK_THRESHOLD = 0.5  # Threshold to invert image (foreground correction)

# LDA Hyperparameters
LDA_SHRINKAGE_CANDIDATES = [0.001, 0.01]  # Fixed shrinkage values for experts
LDA_SOLVER = "lsqr"

# Group A (Robust) Settings
QUANTILE_N_QUANTILES = 50
QUANTILE_OUTPUT_DIST = "normal"

# Group B & C (Polynomial) Settings
POLY_DEGREE = 2
POLY_INTERACTION_ONLY = False  # Full polynomial expansion
POLY_INCLUDE_BIAS = False

# Group C (Pairwise Interaction) Settings
BOTTLENECK_K = 5  # Dimension for discriminative bottleneck projection

# =============================================================================
# 5. EXPERT LIBRARY DEFINITION (PFPGE Architecture)
# =============================================================================
# Defines the library of experts for Dynamic Ensemble Selection.
EXPERTS_CONFIG = {}

# --- Group A: Global Statistical Anchors ---
# Baseline experts using all provided features with robust global transforms.
EXPERTS_CONFIG["A_Marginal"] = {
    "group": "A",
    "features": ["margin", "shape", "texture"],
    "topology": "marginal",  # PowerTransformer -> LDA
}
EXPERTS_CONFIG["A_Rotational"] = {
    "group": "A",
    "features": ["margin", "shape", "texture"],
    "topology": "rotational",  # Power -> PCA(NoWhiten) -> Power -> LDA
}
EXPERTS_CONFIG["A_Robust"] = {
    "group": "A",
    "features": ["margin", "shape", "texture"],
    "topology": "robust",  # QuantileTransformer -> LDA
}

# --- Group B: Physical Polynomial Experts ---
# Domain-specific expert using extracted morphometrics.
EXPERTS_CONFIG["B_MorphoPoly"] = {
    "group": "B",
    "features": ["physical"],
    "topology": "polynomial",  # Power -> Poly(2) -> Power -> LDA
}

# --- Group C: Pairwise-Factorized Interaction Experts ---
# Captures specific biological couplings (e.g., Margin-Texture) via bottlenecking.
# Topology: [Stratified Alignment -> Bottleneck] per group -> Concat -> Poly -> LDA
feature_keys = list(FEATURE_GROUPS.keys())
for i in range(len(feature_keys)):
    for j in range(i + 1, len(feature_keys)):
        f1 = feature_keys[i]
        f2 = feature_keys[j]
        key = f"C_Pairwise_{f1}_{f2}"
        EXPERTS_CONFIG[key] = {
            "group": "C",
            "features": [f1, f2],
            "topology": "pairwise_interaction",
        }
