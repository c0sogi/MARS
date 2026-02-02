import os
import numpy as np

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Specific working directory for this idea to ensure cache isolation
WORKING_DIR = "./working/idea_22"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Global Configuration
# ==========================================
SEED = 42
N_FOLDS = 5
N_JOBS = 12  # Utilizing available 12 vCPUs

# Debugging flags
DEBUG = False
DEBUG_SAMPLE_SIZE = 200  # Subset size for rapid prototyping if DEBUG is True

# ==========================================
# Signal Processing Parameters
# ==========================================
# Dataset properties: 60001 samples over 10 minutes implies ~100 Hz
SAMPLING_RATE = 100

# View 1: Kinematic Trend (Savitzky-Golay)
# Window size > 20 as requested, must be odd.
# Cite solution_lesson_node_00023: Increased window for robust smoothing
SG_WINDOW = 25
SG_POLYORDER = 2  # Quadratic trend extraction

# View 3: Absolute Intensity & View 1/2 Quantiles
# Dense Grid of Quantiles to capture shape distribution
DENSE_QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

# View 4: Structural Spectral Features (PSD via Welch)
# Frequency bands in Hz
PSD_BANDS = {
    "low": (0.5, 3.0),
    "mid": (3.0, 10.0),
    "high": (10.0, 20.0),
    "ultra": (20.0, 45.0),
}

# View 5: Temporal Evolution
# Number of non-overlapping windows to split the signal into for RMS/Mean calculation
TEMPORAL_NUM_WINDOWS = 10

# ==========================================
# Model Hyperparameters
# ==========================================
# Single LightGBM Regressor configuration
# Implements Loss-Metric Decoupling: Optimizes L2 (MSE) for gradient stability, monitors MAE.
LGBM_PARAMS = {
    "objective": "regression",  # L2 Loss (Mean Squared Error)
    "metric": "mae",  # Evaluation metric
    "boosting_type": "gbdt",
    "learning_rate": 0.005,  # Low learning rate (< 0.01) for convergence stability
    "n_estimators": 10000,  # High cap, controlled by early stopping
    "num_leaves": 128,  # Moderate-High complexity
    "max_depth": -1,  # No fixed depth limit
    "reg_alpha": 1.5,  # Explicit L1 Regularization to prevent overfitting
    "reg_lambda": 1.5,  # Explicit L2 Regularization
    "colsample_bytree": 0.7,  # Feature subsampling
    "subsample": 0.7,  # Data subsampling (bagging)
    "subsample_freq": 1,
    "min_child_samples": 50,  # Regularization for leaf nodes
    "random_state": SEED,
    "n_jobs": N_JOBS,
    "verbosity": -1,
}

# Training Control
EARLY_STOPPING_ROUNDS = 150
VERBOSE_EVAL = 100
