import os

# =============================================================================
# Global Configuration & Reproducibility
# =============================================================================
SEED = 42

# =============================================================================
# Directory Paths
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_31"
SUBMISSION_DIR = "./submission"

# Ensure necessary writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# Feature Column Definitions
# =============================================================================
# We define the 192 features (Margin, Shape, Texture) explicitly.
# They are sorted alphanumerically (e.g., 'margin_10' comes before 'margin_2')
# to ensure a deterministic column order across all data loading and processing
# steps, preventing implicit column permutation noise.

_feature_types = ["margin", "shape", "texture"]
_feature_indices = range(1, 65)  # Attributes 1 through 64

# Generate the full list of feature names
_raw_feature_list = [f"{ft}_{i}" for ft in _feature_types for i in _feature_indices]

# Sort alphanumerically to strictly fix the order
FEATURE_COLUMNS = sorted(_raw_feature_list)

# =============================================================================
# Data Processing & Numerical Precision
# =============================================================================
# The solution relies on a Cholesky-based solver which requires high precision.
# We enforce float64 to avoid numerical instability and SVD truncation artifacts
# often found in standard float32 pipelines.
FLOAT_TYPE = "float64"

# Clipping epsilon for log-loss metric calculation
# Predicted probabilities are clipped to [1e-15, 1-1e-15]
CLIP_EPSILON = 1e-15

# =============================================================================
# Hyperparameters & Debugging
# =============================================================================
# Flags to control development iterations and runtime
DEBUG = False
DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True

# Inference parameters
BATCH_SIZE = 1024  # Batch size for prediction (if needed for memory constraints)
