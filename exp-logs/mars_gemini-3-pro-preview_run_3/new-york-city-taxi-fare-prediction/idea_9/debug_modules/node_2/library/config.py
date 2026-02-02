import os
import torch

# ==========================================
# PATHS & DIRECTORIES
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_9"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data Paths (using metadata splits)
PATH_TRAIN = os.path.join(METADATA_DIR, "train.parquet")
PATH_VAL = os.path.join(METADATA_DIR, "val.parquet")
PATH_TEST = os.path.join(METADATA_DIR, "test.parquet")
PATH_SUBMISSION = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# GLOBAL CONSTANTS
# ==========================================
SEED = 42
N_JOBS = 12  # Available vCPUs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# DATA CLEANING & PREPROCESSING
# ==========================================
# Bounding Box (Lat 40–42, Lon -73 to -75)
# Strict filtering to prevent distance explosion
BB_MIN_LAT = 40.0
BB_MAX_LAT = 42.0
BB_MIN_LON = -75.0
BB_MAX_LON = -73.0

# Target Filtering (Fare Amount)
FARE_MIN = 0.0
FARE_MAX = 500.0

# ==========================================
# FEATURE ENGINEERING
# ==========================================
# Rotation for NYC Grid Alignment (29 degrees)
ROTATION_ANGLE = 29

# Landmarks (Latitude, Longitude) for Haversine distance features
LANDMARKS = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "TS": (40.7580, -73.9855),  # Times Square
    "WTC": (40.7126, -74.0099),  # World Trade Center
}

# Spatial Grid Resolution (degrees) for Embeddings
# 0.002 degrees is approx 220m, providing fine-grained spatial context
GRID_RESOLUTION = 0.002

# ==========================================
# DATA SPLITTING
# ==========================================
# 90% Base Train (Level 0), 10% Meta Train (Level 1)
# Applied after loading the full training set
META_TRAIN_SIZE = 0.1

# ==========================================
# MODEL HYPERPARAMETERS
# ==========================================

# 1. XGBoost (GPU, Depth-wise)
# Uses GPU for fast training on deep trees
XGB_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.02,
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "tree_method": "gpu_hist",  # NVIDIA A100
    "predictor": "gpu_predictor",
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "early_stopping_rounds": 100,
}

# 2. LightGBM (CPU, Leaf-wise)
# Uses CPU to diversify from XGBoost, focusing on leaf-wise growth
LGBM_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.02,
    "num_leaves": 128,  # Leaf-wise growth
    "boosting_type": "gbdt",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "regression",
    "metric": "rmse",
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "verbosity": -1,
    "device": "cpu",
    "early_stopping_rounds": 100,
}

# 3. Spatial ResNet (PyTorch)
# Neural Network with Learnable Spatial Embeddings
NN_PARAMS = {
    "batch_size": 4096,  # Large batch size for A100
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 20,
    "embedding_dim": 64,  # Dimension for grid embeddings
    "hidden_dims": [512, 256, 128],
    "dropout": 0.1,
    "early_stopping_patience": 3,
}

# 4. Meta Learner (Ridge Regression)
# Linear stacker to combine base predictions
META_PARAMS = {"alpha": 1.0, "random_state": SEED}

# ==========================================
# DEBUG / SAMPLING
# ==========================================
# Set to an integer (e.g., 100_000) for debugging/fast iteration
# Set to None for full production training (55M rows)
DEBUG_SAMPLE_SIZE = None
