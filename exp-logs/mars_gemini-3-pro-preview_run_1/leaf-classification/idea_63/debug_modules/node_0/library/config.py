import os

# ==========================================
# Global Configuration
# ==========================================

# Random Seed for reproducibility across numpy, torch, and sklearn
SEED = 42

# ==========================================
# Directory Paths
# ==========================================

# Read-only input directory containing raw data and images
INPUT_DIR = "./input"

# Directory containing generated metadata CSVs (train.csv, val.csv, test.csv)
METADATA_DIR = "./metadata"

# Working directory for intermediate files
WORKING_DIR = "./working"

# Specific cache directory for this experimental run (Idea 63)
# Used to store processed features (parquet/npy) to avoid re-computation
CACHE_DIR = os.path.join(WORKING_DIR, "idea_63")

# Directory for final submission output
SUBMISSION_DIR = "./submission"

# ==========================================
# File Paths
# ==========================================

# Metadata file paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Final submission file path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Feature Configuration
# ==========================================

# List of geometric features to be extracted from binary masks.
# These keys are used for dictionary-based assembly to prevent positional indexing errors.
GEOMETRIC_FEATURES = [
    "Area",  # Integral measure of size
    "Mean_Thickness",  # Internal structure (Euclidean Distance Transform mean)
    "Eccentricity",  # Elongation via ellipse fitting
    "Solidity",  # Roughness (Area / ConvexArea)
    "Extent",  # Rectangularity (Area / BoundingRectArea)
    "Aspect_Ratio",  # Orientation (BoundingWidth / BoundingHeight)
]
