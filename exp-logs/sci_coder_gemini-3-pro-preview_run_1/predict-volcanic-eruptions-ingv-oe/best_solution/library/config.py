import os
import torch

# ==========================================
# 1. PATHS & DIRECTORIES
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Working directory for Idea 13 (Sub-band Energy & Crest Factor)
WORK_DIR = "./working/idea_13"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Cache Paths for Deterministic Processing
TABULAR_TRAIN_CACHE = os.path.join(WORK_DIR, "train_tabular_features.parquet")
TABULAR_VAL_CACHE = os.path.join(WORK_DIR, "val_tabular_features.parquet")
TABULAR_TEST_CACHE = os.path.join(WORK_DIR, "test_tabular_features.parquet")

VISION_GLOBAL_MAX_CACHE = os.path.join(WORK_DIR, "vision_global_max.npy")
VISION_TRAIN_CACHE_DIR = os.path.join(WORK_DIR, "spectrograms_train")
VISION_VAL_CACHE_DIR = os.path.join(WORK_DIR, "spectrograms_val")
VISION_TEST_CACHE_DIR = os.path.join(WORK_DIR, "spectrograms_test")

# ==========================================
# 2. GLOBAL SETTINGS
# ==========================================
SEED = 42
N_FOLDS = 5
DEBUG = False  # Set to True to run on a small subset for testing
MAX_DEBUG_SAMPLES = 100  # Number of samples to use if DEBUG is True

# ==========================================
# 3. TABULAR BRANCH SETTINGS (Branch A)
# ==========================================
# Feature Engineering
MFCC_COEFFS = 13  # Keep low-order coefficients (1-13) to capture timbre without noise
DROP_SPECTRAL_MIN_MAX = True  # Multi-View Policy: Drop Min/Max for spectral features

# LightGBM Hyperparameters
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "n_estimators": 10000,
    "learning_rate": 0.01,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "n_jobs": 12,
    "random_state": SEED,
    "verbose": -1,
    "early_stopping_rounds": 200,
}

# ==========================================
# 4. VISION BRANCH SETTINGS (Branch B)
# ==========================================
# Signal Processing / Spectrograms
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 224  # Matches image height
FMIN = 20
FMAX = 20000
IMG_SIZE = (224, 224)  # (Height, Width) for EfficientNet-B0
IN_CHANNELS = 10  # One channel per sensor

# Normalization & Scaling
USE_LOG_TARGET = True  # Apply np.log1p to target for Vision branch
GLOBAL_MAX_SAMPLE_SIZE = 500  # Number of samples to estimate global max energy

# Model Architecture
MODEL_NAME = "efficientnet_b0"
PRETRAINED = True

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 35  # Extended training for convergence
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 5. META-LEARNER SETTINGS
# ==========================================
META_MODEL_ALPHA = 1.0  # Ridge Regression Alpha
