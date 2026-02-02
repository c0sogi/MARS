import os
import numpy as np
import torch
import random

# ====================================================
# PATH CONFIGURATION
# ====================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Working and Output Directories
CACHE_DIR = "./working/idea_8"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ====================================================
# REPRODUCIBILITY
# ====================================================
SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ====================================================
# DATA PROCESSING HYPERPARAMETERS
# ====================================================
# Image Parameters
IMG_SIZE = 224
SLICES_PER_ZONE = 1  # 1 slice for Apex, 1 for Mid, 1 for Base
TOTAL_SLICES = 3  # Total slices extracted per patient for the CNN

# Hounsfield Unit (HU) Windowing for CNN Input
# Standard Lung Window
HU_MIN = -1000
HU_MAX = 400

# Volumetric Feature Calculation Thresholds
# Tissues between these values are considered lung parenchyma
VOL_HU_MIN = -1000
VOL_HU_MAX = -400

# ====================================================
# MODEL HYPERPARAMETERS
# ====================================================
# Dimensionality Reduction
N_PCA_COMPONENTS = 40

# Quantile Regression
QUANTILE_ALPHA = 0.5  # Optimization target for FVC (Median)

# Metric / Post-processing
MIN_CONFIDENCE = 70  # Lower bound for sigma
MAX_ERROR_METRIC = 1000  # Error threshold in metric calculation

# ====================================================
# TRAINING HYPERPARAMETERS
# ====================================================
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOPPING_ROUNDS = 10

# Feature Names (for consistency across modules)
# These will be generated during feature engineering
TABULAR_COLS = [
    "Age",
    "Sex",
    "SmokingStatus",
    "Baseline_FVC",
    "Baseline_Percent",
    "Baseline_Weeks",
]
