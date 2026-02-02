import os

# ====================================================
# Global Configuration & Reproducibility
# ====================================================
SEED = 42

# ====================================================
# Directory & File Paths
# ====================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"

# Metadata Files (Pre-generated)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Submission Template
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Directory for Idea 15 (Deterministic Data Processing)
CACHE_DIR = os.path.join(WORKING_DIR, "idea_15")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ====================================================
# Image Processing & Volumetrics
# ====================================================
# Input size for EfficientNet-B0
IMAGE_SIZE = 224

# Multi-Axis Variance Sampling
# We select the top N slices with highest variance from both Axial and Coronal planes
SLICES_PER_AXIS = 3
TOTAL_SLICES = SLICES_PER_AXIS * 2  # 3 Axial + 3 Coronal

# Hounsfield Unit (HU) Thresholds for Lung Masking
HU_MIN = -1000
HU_MAX = -400

# ====================================================
# Feature Extraction & Dimensionality Reduction
# ====================================================
# Number of components to keep after PCA on concatenated features
PCA_COMPONENTS = 40

# Number of bins for Density Histogram (Emphysema, Healthy, Fibrosis, Consolidation)
HISTOGRAM_BINS = 4

# ====================================================
# Model Hyperparameters
# ====================================================
# 1. FVC Predictor: Linear Quantile Regressor
# We target the Median (q=0.5) to minimize L1 loss (Laplace Metric alignment)
QUANTILE = 0.5

# 2. Uncertainty Predictor: Elastic Net Regressor
# Predicts absolute residuals (MAD)
# Alphas for regularization strength (can be used in CV)
ELASTIC_NET_ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0]
ELASTIC_NET_L1_RATIO = 0.5  # Mix between L1 and L2

# ====================================================
# Metric & Post-Processing Constants
# ====================================================
# Constants for the modified Laplace Log Likelihood metric
MIN_CONFIDENCE = 70  # sigma_clipped
MAX_ERROR = 1000  # Delta clipping

# ====================================================
# Compute Resources
# ====================================================
NUM_WORKERS = 2
N_JOBS = 12  # Utilize available vCPUs for sklearn
