import os

# ====================================================
# DIRECTORY PATHS
# ====================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Create working and submission directories if they don't exist
# (Safe to do here as this config is imported by all other modules)
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Idea-specific directory for caching
IDEA_NAME = "idea_1"
CACHE_DIR = os.path.join(WORKING_DIR, IDEA_NAME)
os.makedirs(CACHE_DIR, exist_ok=True)

# ====================================================
# FILE PATHS
# ====================================================
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache file for extracted radiomics features
RADIOMICS_CACHE_PATH = os.path.join(CACHE_DIR, "radiomics_features.parquet")

# ====================================================
# IMAGE PROCESSING (RADIOMICS) PARAMETERS
# ====================================================
# Hounsfield Unit thresholds for lung tissue segmentation
HU_MIN = -1000
HU_MAX = -400

# ====================================================
# MODEL HYPERPARAMETERS (ELASTIC NET)
# ====================================================
# Alpha: Constant that multiplies the penalty terms.
# L1_ratio: The ElasticNet mixing parameter (0=L2, 1=L1).
ELASTIC_NET_ALPHA = 1.0
ELASTIC_NET_L1_RATIO = 0.5
MAX_ITER = 5000

# Random Seed for reproducibility
RANDOM_STATE = 42

# ====================================================
# FEATURE ENGINEERING
# ====================================================
# Features extracted from CT scans
RADIOMICS_FEATURES = ["Lung_Volume", "Mean_Density", "Density_Variance"]

# Continuous tabular features to be normalized
# Note: 'Weeks' refers to relative weeks from baseline
TABULAR_NUMERICAL_FEATURES = ["Weeks", "Age", "Baseline_FVC", "Baseline_Percent"]

# Categorical tabular features to be One-Hot Encoded
TABULAR_CATEGORICAL_FEATURES = ["Sex", "SmokingStatus"]

# Target columns
TARGET_COL = "FVC"
CONFIDENCE_COL = "Confidence"

# ====================================================
# METRIC & EVALUATION CONSTANTS
# ====================================================
# Constants for the modified Laplace Log Likelihood metric
MIN_CONFIDENCE = 70  # sigma_clipped
MAX_ERROR = 1000  # Delta threshold
