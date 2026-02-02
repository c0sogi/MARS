import os
import torch

# ==========================================
# 1. PATHS & DIRECTORIES
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_8"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# File Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = "./submission/submission.csv"

# Cache Paths
CACHE_TRAIN_TREE = os.path.join(WORKING_DIR, "train_tree.parquet")
CACHE_VAL_TREE = os.path.join(WORKING_DIR, "val_tree.parquet")
CACHE_TEST_TREE = os.path.join(WORKING_DIR, "test_tree.parquet")

CACHE_TRAIN_NN = os.path.join(WORKING_DIR, "train_nn.parquet")
CACHE_VAL_NN = os.path.join(WORKING_DIR, "val_nn.parquet")
CACHE_TEST_NN = os.path.join(WORKING_DIR, "test_nn.parquet")

CACHE_SCALER = os.path.join(WORKING_DIR, "scaler.joblib")

# Model Paths
MODEL_XGB_PATH = os.path.join(WORKING_DIR, "xgboost_model.json")
MODEL_LGBM_PATH = os.path.join(WORKING_DIR, "lgbm_model.txt")
MODEL_RESNET_PATH = os.path.join(WORKING_DIR, "resnet_model.pth")
MODEL_META_PATH = os.path.join(WORKING_DIR, "meta_ridge.joblib")

# ==========================================
# 2. GLOBAL CONSTANTS & SEEDS
# ==========================================
SEED = 42
N_JOBS = 12  # Available vCPUs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data Filtering / Sanitation
# Bounding box for NYC (Lat 40-42, Lon -73 to -75)
MIN_LAT = 40.0
MAX_LAT = 42.0
MIN_LON = -75.0
MAX_LON = -73.0

# Fare Amount Filter
MIN_FARE = 0.0
MAX_FARE = 500.0

# Coordinate Rotation (29 degrees for NYC grid alignment)
ROTATION_ANGLE = 29

# Landmarks (Lat, Lon)
LANDMARKS = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "EWR": (40.6895, -74.1745),
    "TSQ": (40.7580, -73.9855),
    "WTC": (40.7118, -74.0131),
    "MET": (40.8135, -74.0745),
}

# ==========================================
# 3. FEATURE LISTS
# ==========================================

# Base raw features available in dataset
BASE_FEATURES = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

# Features common to both pipelines (calculated)
SHARED_ENGINEERED_FEATURES = [
    "abs_diff_longitude",
    "abs_diff_latitude",
    "haversine_dist",
    "manhattan_dist",
    "euclidean_dist",
    "pickup_rot_x",
    "pickup_rot_y",
    "dropoff_rot_x",
    "dropoff_rot_y",
] + [f"dist_to_{k}" for k in LANDMARKS.keys()]

# --- Tree-Specific Features ---
# Trees handle raw integers well. No scaling needed.
TREE_TIME_FEATURES = ["hour", "day", "month", "year", "day_of_week"]

TREE_FEATURES = BASE_FEATURES + SHARED_ENGINEERED_FEATURES + TREE_TIME_FEATURES

# --- Neural Network Specific Features ---
# NN needs scaled inputs and cyclical encodings for time.
# Continuous features to be StandardScaled
NN_CONTINUOUS_FEATURES = (
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    + SHARED_ENGINEERED_FEATURES
    + ["year"]  # Treat year as continuous/ordinal
)

# Cyclical features (Sin/Cos transform)
NN_CYCLICAL_FEATURES = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]

# Categorical features for Entity Embeddings
# (Name, Num_Classes, Embedding_Dim)
NN_EMBEDDING_CONFIG = [
    ("passenger_count", 10, 4),  # Cap at 9 passengers?
    # day_of_week is handled via sin/cos in this strategy per prompt,
    # but prompt also mentions "Entity Embeddings for categorical features (e.g., Day of Week)".
    # We will stick to the prompt's explicit instruction to use Sin/Cos for Hour/Month
    # and use Embedding for Day of Week as an example of hybrid approach.
    ("day_of_week", 7, 3),
]

# Remove day_of_week sin/cos if we use embedding, or keep both?
# The prompt says: "Transform Hour and Month into Sine and Cosine... Entity Embeddings for... Day of Week".
# So we adjust the lists:
NN_CYCLICAL_FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos"]

# Final dense input size calculation needed for model def
# Continuous + Cyclical
NN_DENSE_INPUT_DIM = len(NN_CONTINUOUS_FEATURES) + len(NN_CYCLICAL_FEATURES)

# ==========================================
# 4. MODEL HYPERPARAMETERS
# ==========================================

# XGBoost (GPU) - Depth-wise
XGB_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": 10,  # Deep trees
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "tree_method": "gpu_hist",  # GPU acceleration
    "gpu_id": 0,
    "random_state": SEED,
    "n_jobs": N_JOBS,
    "predictor": "gpu_predictor",
}

# LightGBM (CPU) - Leaf-wise
LGBM_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "num_leaves": 128,  # Leaf-wise growth
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "device": "cpu",
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "verbose": -1,
}

# Deep Spatial ResNet (PyTorch)
RESNET_PARAMS = {
    "batch_size": 4096,
    "learning_rate": 1e-3,
    "epochs": 15,
    "patience": 3,
    "hidden_dims": [512, 256, 128, 64],
    "dropout": 0.1,
    "weight_decay": 1e-5,
}

# Meta Learner (Ridge)
META_PARAMS = {"alpha": 1.0, "random_state": SEED}

# ==========================================
# 5. TRAINING CONFIGURATION
# ==========================================
# Split for Stacking
# 90% Base Train, 10% Meta Train (Hold-out blending)
META_TRAIN_SIZE = 0.10

# Debugging / Sampling
# Set to None to use full dataset, or an integer to sample
DEBUG_SAMPLE_SIZE = None
