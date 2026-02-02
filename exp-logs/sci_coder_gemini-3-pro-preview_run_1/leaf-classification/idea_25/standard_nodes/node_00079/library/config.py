import numpy as np
import os

# ==========================================
# Global Reproducibility & Precision
# ==========================================
SEED = 42
# We use float64 to minimize numerical instability and avoid the float32 machine epsilon floor
# when calculating probabilities close to 0 or 1.
FLOAT_PRECISION = np.float64

# ==========================================
# Directory Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Specific working directory for this idea iteration
WORKING_DIR = "./working/idea_25"
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Data Schema & Features
# ==========================================
ID_COLUMN = "id"
TARGET_COLUMN = "species"

# Hardcoded feature generation to ensure deterministic ordering.
# The dataset contains 3 types of features, each with a 64-attribute vector.
_margin_cols = [f"margin{i}" for i in range(1, 65)]
_shape_cols = [f"shape{i}" for i in range(1, 65)]
_texture_cols = [f"texture{i}" for i in range(1, 65)]

# Final combined feature list (192 features)
FEATURE_COLUMNS = _margin_cols + _shape_cols + _texture_cols

# ==========================================
# Hyperparameters & Constants
# ==========================================
# Clip epsilon for log loss calculation as defined in the metric description
CLIP_EPSILON = 1e-15

# Number of CPU cores available for parallel processing
N_JOBS = 12

# Default hyperparameters for the pipeline
# These can be overridden in the training script but serve as defaults here.
CONFIG = {
    "random_state": SEED,
    "test_size": 0.2,
    "n_jobs": N_JOBS,
}
