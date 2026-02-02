import os

# -----------------------------------------------------------------------------
# Global Directory Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Specific working directory for Idea 20 (Algebraically-Stabilized OAS)
WORKING_DIR = "./working/idea_20"

# Ensure the working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Global Configuration
# -----------------------------------------------------------------------------
SEED = 42

# -----------------------------------------------------------------------------
# Column Definitions
# -----------------------------------------------------------------------------
ID_COL = "id"
TARGET_COL = "species"

# -----------------------------------------------------------------------------
# Feature Columns
# -----------------------------------------------------------------------------
# The dataset provides three sets of features, each with 64 attributes.
# We generate the names explicitly and sort them alphanumerically.
# This strictly enforces a deterministic column order for the 192 features
# to prevent alignment errors between training and inference.

_margin_cols = [f"margin{i}" for i in range(1, 65)]
_shape_cols = [f"shape{i}" for i in range(1, 65)]
_texture_cols = [f"texture{i}" for i in range(1, 65)]

# Combine all feature names
_all_features = _margin_cols + _shape_cols + _texture_cols

# Sort alphanumerically (e.g., margin1, margin10, margin11, ..., margin2)
FEATURE_COLS = sorted(_all_features)

# Sanity check to ensure exactly 192 features are defined
assert len(FEATURE_COLS) == 192, f"Expected 192 features, got {len(FEATURE_COLS)}"
