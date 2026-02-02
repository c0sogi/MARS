import os

# ==========================================
# Global Configuration
# ==========================================
SEED = 42
N_FOLDS = 5
TARGET_COL = "Cover_Type"
ID_COL = "Id"

# Class Mapping: The dataset contains classes [1, 2, 3, 4, 6, 7].
# We map them to [0, 1, 2, 3, 4, 5] for training.
CLASS_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4, 7: 5}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
NUM_CLASSES = len(CLASS_MAP)

# ==========================================
# File Paths
# ==========================================
# Input Metadata (Generated in previous steps)
INPUT_DIR = "./metadata"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.parquet")
VAL_PATH = os.path.join(INPUT_DIR, "val.parquet")
TEST_PATH = os.path.join(INPUT_DIR, "test.parquet")

# Output Directories
WORKING_DIR = "./working/idea_2"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
MODEL_DIR = os.path.join(WORKING_DIR, "models")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Feature Engineering Configuration
# ==========================================
# Interaction pairs to synthesize: (Continuous Feature, Binary/Categorical Feature)
# These are designed to capture specific sub-domain behaviors (e.g., Elevation effects within specific Wilderness Areas).
INTERACTION_PAIRS = [
    ("Elevation", "Wilderness_Area1"),
    ("Elevation", "Wilderness_Area2"),
    ("Elevation", "Wilderness_Area3"),
    ("Elevation", "Wilderness_Area4"),
    ("Horizontal_Distance_To_Hydrology", "Wilderness_Area1"),
    ("Horizontal_Distance_To_Roadways", "Wilderness_Area1"),
    ("Horizontal_Distance_To_Fire_Points", "Wilderness_Area1"),
]

# ==========================================
# Model Hyperparameters
# ==========================================

# --- Level-0: LightGBM ---
LGBM_PARAMS = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_class": NUM_CLASSES,
    "learning_rate": 0.05,
    "num_leaves": 128,
    "max_depth": -1,  # No limit
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 0.5,
    "n_estimators": 3000,
    "early_stopping_rounds": 100,
    "verbose": -1,
    "n_jobs": -1,
    "seed": SEED,
    "verbosity": -1,
}

# --- Level-0: Tabular Neural Network (ResNet-MLP style) ---
NN_PARAMS = {
    "input_dim": None,  # To be set dynamically based on feature count
    "output_dim": NUM_CLASSES,
    "hidden_layers": [512, 256, 128],
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 2048,
    "epochs": 40,
    "early_stopping_patience": 7,
    "scheduler_factor": 0.2,
    "scheduler_patience": 3,
    "seed": SEED,
}

# --- Level-1: Meta Learner (Logistic Regression) ---
META_PARAMS = {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": SEED,
    "n_jobs": -1,
}

# ==========================================
# Champion-Challenger Guard
# ==========================================
# The ensemble must beat this validation accuracy score to be accepted.
# Set based on previous best single-model performance (approximate).
BASELINE_SCORE = 0.95
