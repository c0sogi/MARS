import os
import numpy as np

# =============================================================================
# Directories and File Paths
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_41")
SUBMISSION_DIR = "./submission"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Metadata Files
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# Global Configuration & Hyperparameters
# =============================================================================
SEED = 42
# Strict requirement for Double Precision to avoid metric floors
PRECISION_TYPE = np.float64

# =============================================================================
# Feature Definitions
# =============================================================================
# 1. Geometric Features (Visual) - To be extracted from binary images
# Capturing Scale, Morphology, and Topology
GEOMETRIC_FEATURES = [
    "Area",
    "Perimeter",
    "Convex_Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Solidity",
    "Eccentricity",
    "Min_Area_Aspect_Ratio",
    "Extent",
    "Convexity",
]

# 2. Tabular Feature Prefixes (Provided in dataset)
TABULAR_PREFIXES = ["margin", "shape", "texture"]
NUM_TABULAR_FEATURES_PER_GROUP = 64

# 3. Column Names
ID_COL = "id"
TARGET_COL = "species"
FILE_PATH_COL = "file_path"


# =============================================================================
# Utility Functions
# =============================================================================
def setup_directories():
    """
    Ensures that the working, cache, and submission directories exist.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)


def get_all_feature_names():
    """
    Returns the complete list of feature names (Tabular + Geometric),
    sorted alphanumerically to ensure deterministic memory layout and
    floating-point associativity.
    """
    # Generate tabular feature names (e.g., margin_1, margin_2...)
    tabular_features = []
    for prefix in TABULAR_PREFIXES:
        for i in range(1, NUM_TABULAR_FEATURES_PER_GROUP + 1):
            tabular_features.append(f"{prefix}_{i}")

    # Combine with geometric features
    all_features = tabular_features + GEOMETRIC_FEATURES

    # Enforce alphanumeric sorting
    return sorted(all_features)
