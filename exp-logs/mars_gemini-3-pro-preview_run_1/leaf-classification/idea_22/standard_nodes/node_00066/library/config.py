import os

# ==========================================
# Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
OUTPUT_DIR = "./submission"
WORKING_DIR = "./working/idea_22"

# Ensure necessary directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# ==========================================
# Reproducibility
# ==========================================
RANDOM_SEED = 42

# ==========================================
# Feature Definition
# ==========================================
# We define the 192 features derived from margin, shape, and texture.
# We sort them alphanumerically (standard string sort) to enforce a
# deterministic schema (e.g., margin1, margin10, margin11, ...).

_margin_features = [f"margin{i}" for i in range(1, 65)]
_shape_features = [f"shape{i}" for i in range(1, 65)]
_texture_features = [f"texture{i}" for i in range(1, 65)]

FEATURES = sorted(_margin_features + _shape_features + _texture_features)
