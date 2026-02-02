import os
import torch
import numpy as np
import random

# ==========================================
# 1. PATHS & DIRECTORIES
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = "./working/idea_7"
SUBMISSION_DIR = "./submission"

# Ensure writeable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# 2. GLOBAL SETTINGS & REPRODUCIBILITY
# ==========================================
SEED = 42
DEBUG = False
DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True


def seed_everything(seed=42):
    """Sets the random seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 3. DATA PARAMETERS
# ==========================================
NUM_SENSORS = 10
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]
SEGMENT_LENGTH = 60001
SAMPLING_RATE = 100  # 100 Hz (60000 samples in 10 minutes)

# ==========================================
# 4. FEATURE EXTRACTION (TABULAR)
# ==========================================
# MFCC Parameters for Parsimonious Cepstral Features
MFCC_SAMPLE_RATE = SAMPLING_RATE
MFCC_N_MFCC = 13  # Low-order coefficients only
MFCC_N_FFT = 1024
MFCC_HOP_LENGTH = 512
MFCC_N_MELS = 40

# ==========================================
# 5. SPECTROGRAM PARAMETERS (VISION)
# ==========================================
# Parameters for Log-Mel Spectrogram generation
SPEC_N_FFT = 256
SPEC_HOP_LENGTH = 64
SPEC_N_MELS = 128
SPEC_FMIN = 0
SPEC_FMAX = None  # Defaults to Nyquist (50 Hz)

# ==========================================
# 6. MODEL & TRAINING PARAMETERS
# ==========================================
NUM_FOLDS = 5
BATCH_SIZE = 32
CNN_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
PATIENCE = 5  # For scheduler or early stopping

# LightGBM Hyperparameters
LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": SEED,
    "n_jobs": -1,
}
LGB_ROUNDS = 5000
LGB_EARLY_STOPPING_ROUNDS = 100

# Meta-Learner (Ridge) Parameters
RIDGE_ALPHA = 1.0

# ==========================================
# 7. HARDWARE
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
