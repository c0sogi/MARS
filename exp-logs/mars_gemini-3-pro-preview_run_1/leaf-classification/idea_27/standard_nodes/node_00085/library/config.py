import os

# -----------------------------------------------------------------------------
# Global Directory Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
IDEA_DIR = os.path.join(WORKING_DIR, "idea_28")

# Ensure the working directory for this idea exists
os.makedirs(IDEA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Global Configuration Constants
# -----------------------------------------------------------------------------
SEED = 42
TARGET_COL = "species"

# -----------------------------------------------------------------------------
# Feature Definitions
# -----------------------------------------------------------------------------
# The dataset contains 3 sets of features, each with 64 attributes.
# We define them here to ensure a deterministic order throughout the pipeline.
FEATURE_COLS = []
_feature_types = ["margin", "shape", "texture"]

for _ftype in _feature_types:
    for _i in range(1, 65):
        FEATURE_COLS.append(f"{_ftype}{_i}")

# Verify we have exactly 192 features
assert len(FEATURE_COLS) == 192, f"Expected 192 features, got {len(FEATURE_COLS)}"
