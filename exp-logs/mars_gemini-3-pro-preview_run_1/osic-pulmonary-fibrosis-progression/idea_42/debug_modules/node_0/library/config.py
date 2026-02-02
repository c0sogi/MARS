import os
import torch

# ==========================================
# 1. File System Paths
# ==========================================
INPUT_PATH = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"

# Cache Directory for Deterministic Data Processing
# As per requirements, we ensure this directory exists immediately
CACHE_DIR = os.path.join(WORKING_DIR, "idea_42")
os.makedirs(CACHE_DIR, exist_ok=True)

# DICOM Directories
TRAIN_DICOM_DIR = os.path.join(INPUT_PATH, "train")
TEST_DICOM_DIR = os.path.join(INPUT_PATH, "test")

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_PATH, "sample_submission.csv")

# ==========================================
# 2. Model Hyperparameters (CR-HDAN)
# ==========================================
# Using EfficientNet-B0 as per "Low-Capacity Regime" lesson
MODEL_NAME = "tf_efficientnet_b0_ns"

# Native resolution for B0 to avoid overfitting
IMAGE_SIZE = 224

# Native output dimension of B0 (Global Average Pooling)
# We do not compress this to preserve texture signals
EMBED_DIM = 1280

# Dimensionality for the projection/alignment layers
HIDDEN_DIM = 1280

# Dropout rate for regularization in the FFN
DROPOUT = 0.5

# ==========================================
# 3. Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50
PATIENCE = 8  # Strict early stopping
NUM_WORKERS = 4
SEED = 42

# ==========================================
# 4. Feature Definitions
# ==========================================
# The target variable to predict
TARGET_COL = "FVC"

# Tabular features used in the "Shallow Non-Linear Tabular Alignment"
# Note: 'Percent' is included as a strong prior
TABULAR_COLS = ["Age", "Sex", "SmokingStatus", "Percent"]

# Sub-lists for preprocessing logic
NUMERICAL_COLS = ["Age", "Percent"]
CATEGORICAL_COLS = ["Sex", "SmokingStatus"]

# ==========================================
# 5. Metric & Inference Constants
# ==========================================
# Thresholds for the modified Laplace Log Likelihood
MAX_ERROR = 1000.0
MIN_CONFIDENCE = 70.0

# ==========================================
# 6. Debugging & Environment
# ==========================================
# Flags to control dataset size for rapid prototyping
DEBUG = False
DEBUG_DATASET_SIZE = 50

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
