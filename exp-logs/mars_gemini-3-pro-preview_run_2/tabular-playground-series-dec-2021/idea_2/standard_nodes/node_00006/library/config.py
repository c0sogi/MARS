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

# --- LightGBM Configuration ---
# Optimized for full dataset training (Cite solution_lesson_node_00004)
LGBM_PARAMS = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_class": NUM_CLASSES,
    "learning_rate": 0.05,
    "num_leaves": 256,  # Increased capacity for large data
    "max_depth": -1,
    "min_data_in_leaf": 100,  # Increased to reduce overfitting on granular splits
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "n_estimators": 3000,
    "early_stopping_rounds": 100,
    "verbose": -1,
    "n_jobs": -1,
    "seed": SEED,
    "verbosity": -1,
}

# ==========================================
# Champion-Challenger Guard
# ==========================================
# The ensemble must beat this validation accuracy score to be accepted.
# Set based on previous best single-model performance (approximate).
BASELINE_SCORE = 0.95
