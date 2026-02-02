import os
import torch

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Random Seed for reproducibility
SEED = 42

# Compute Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Number of vCPUs available
NUM_WORKERS = 12

# =============================================================================
# DIRECTORY PATHS
# =============================================================================

# Read-only input directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Working directory for Idea 18 (Caching embeddings and models)
WORKING_DIR = "./working/idea_18"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Metadata File Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# =============================================================================
# STREAM A CONFIGURATION (ConvNeXt-Large)
# =============================================================================
# Role: Baseline, Texture-biased, 224px resolution
# Weights: IMAGENET1K_V1 (Standard ImageNet-1k)

STREAM_A = {
    "name": "stream_a",
    "model_name": "convnext_large",
    "weights": "IMAGENET1K_V1",
    "input_size": 224,
    "interpolation": "bicubic",  # ConvNeXt uses bicubic
    "batch_size": 32,
    "embedding_dim": 1536,  # Feature dimension for ConvNeXt-Large
    # Multi-View Generation Parameters
    "views": {
        # Global: Resize to Input Size (Squish). Captures topology.
        "global": {"resize_dims": (224, 224), "crop_size": None},
        # Standard: Resize and Center Crop. Matches pre-training (Resize 232 -> Crop 224 for ConvNeXt).
        "standard": {"resize_size": 232, "crop_size": 224},
        # Local: Resize to 1.28x Input Size and Center Crop (Zoom). Captures texture.
        "local": {"resize_size": int(224 * 1.28), "crop_size": 224},  # ~286
    },
}

# =============================================================================
# STREAM B CONFIGURATION (RegNetY-128GF)
# =============================================================================
# Role: High-Capacity, Weakly-Supervised (SWAG), 384px resolution
# Weights: IMAGENET1K_SWAG_E2E_V1 (Billions of weakly supervised images)

STREAM_B = {
    "name": "stream_b",
    "model_name": "regnet_y_128gf",
    "weights": "IMAGENET1K_SWAG_E2E_V1",
    "input_size": 384,
    "interpolation": "bicubic",
    "batch_size": 16,  # Massive model, smaller batch size
    "embedding_dim": 7392,  # Feature dimension for RegNetY-128GF
    # Multi-View Generation Parameters
    "views": {
        # Global: Resize to Input Size (Squish).
        "global": {"resize_dims": (384, 384), "crop_size": None},
        # Standard: Resize and Center Crop. SWAG 384 usually implies direct resize or slight crop.
        # We use a slight upscale for resize to allow for a clean center crop.
        "standard": {"resize_size": 384, "crop_size": 384},
        # Local: Resize to 1.28x Input Size and Center Crop (Zoom).
        "local": {"resize_size": int(384 * 1.28), "crop_size": 384},  # ~491
    },
}

# =============================================================================
# ENSEMBLE & TRAINING CONFIGURATION
# =============================================================================

ENSEMBLE = {
    # Logistic Regression Head Parameters
    "cv_folds": 5,  # Cross-validation folds for auto-tuning C
    "max_iter": 2000,  # Max iterations for solver convergence
    "n_jobs": -1,  # Use all cores
    "random_state": SEED,
    # Optimization
    "metric": "log_loss",
}

# Cache Settings
CACHE_DIR = WORKING_DIR
LOAD_CACHED_DATA = True  # Set to False to force re-computation of embeddings
