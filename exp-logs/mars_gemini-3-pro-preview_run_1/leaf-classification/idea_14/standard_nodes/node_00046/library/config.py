import os
import numpy as np

# ==========================================
# Global Configuration
# ==========================================
SEED = 42
FLOAT_PRECISION = np.float32

# ==========================================
# Directory and File Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_14")
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Dataset Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Schema
# ==========================================
ID_COL = "id"
TARGET_COL = "species"

# Explicitly define the 192 features in the specific order: Margin -> Shape -> Texture
# This ensures deterministic ordering and prevents implicit permutation by solvers.
FEATURES = []
for i in range(1, 65):
    FEATURES.append(f"margin{i}")
for i in range(1, 65):
    FEATURES.append(f"shape{i}")
for i in range(1, 65):
    FEATURES.append(f"texture{i}")

# Verify feature count
assert len(FEATURES) == 192, f"Expected 192 features, got {len(FEATURES)}"
