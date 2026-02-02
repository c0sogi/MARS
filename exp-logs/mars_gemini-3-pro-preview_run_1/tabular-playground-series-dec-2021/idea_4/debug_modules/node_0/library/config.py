import os

# --- Directory Setup ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_4")

# Create necessary directories for outputs and cache
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Data Paths ---
DATA_PATHS = {
    "train_path": os.path.join(METADATA_DIR, "train.csv"),
    "val_path": os.path.join(METADATA_DIR, "val.csv"),
    "test_path": os.path.join(METADATA_DIR, "test.csv"),
    "sample_submission": os.path.join(INPUT_DIR, "sample_submission.csv"),
    "submission_output": os.path.join(SUBMISSION_DIR, "submission.csv"),
    "cache_dir": CACHE_DIR,
}

# --- Column Definitions ---
TARGET_COL = "Cover_Type"
ID_COL = "Id"

# --- Model Parameters ---
# XGBoost configuration based on the strategy:
# - GPU acceleration (device='cuda', tree_method='hist')
# - High capacity (max_depth=10)
# - Controlled learning rate (eta=0.05)
MODEL_PARAMS = {
    "n_estimators": 3000,  # Sufficiently high to allow early stopping to trigger
    "learning_rate": 0.05,  # eta
    "max_depth": 10,
    "tree_method": "hist",
    "device": "cuda",  # Use NVIDIA A100
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "n_jobs": 12,
    "random_state": 42,
    "verbosity": 0,  # Silent mode
}

# --- Pipeline Parameters ---
PIPELINE_PARAMS = {
    "n_folds": 5,
    "pseudo_label_threshold": 0.99,
    "random_state": 42,
    "early_stopping_rounds": 50,
    "verbose_eval": 100,  # Log metrics every 100 rounds
    "use_geometry_features": True,
}
