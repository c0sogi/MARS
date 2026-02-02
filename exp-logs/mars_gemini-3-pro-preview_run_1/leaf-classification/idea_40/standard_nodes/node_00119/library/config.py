import os
import numpy as np

# ==========================================
# Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_40"
SUBMISSION_DIR = "./submission"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Ensure mutable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# ==========================================
# Global Configuration
# ==========================================
SEED = 42
FLOAT_PRECISION = np.float64

# ==========================================
# Feature Engineering Hyperparameters
# ==========================================
# Spectral Features: Elliptical Fourier Descriptors
# We compute the first 15 harmonics (60 coefficients)
EFD_HARMONICS = 15

# Spatial Features: Macro-Geometric Scalars
# Extracted from binary masks to capture physical size and gross shape
SPATIAL_FEATURES = [
    "Area",
    "Perimeter",
    "Major_Axis",
    "Minor_Axis",
    "Solidity",
    "Extent",
    "Aspect_Ratio",
    "Equivalent_Diameter",
]

# Tabular Feature Configuration
# Pre-extracted features provided in the dataset
NUM_MARGIN_FEATURES = 64
NUM_SHAPE_FEATURES = 64
NUM_TEXTURE_FEATURES = 64

MARGIN_PREFIX = "margin"
SHAPE_PREFIX = "shape"
TEXTURE_PREFIX = "texture"

# Image Processing
# Images are black leaves on white background.
BINARY_THRESHOLD = 127
