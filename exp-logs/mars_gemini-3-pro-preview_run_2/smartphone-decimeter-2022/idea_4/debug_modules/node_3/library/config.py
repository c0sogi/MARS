import os

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_4"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Cache File Paths
CACHE_FILES = {
    "train_data": os.path.join(WORKING_DIR, "train_data.parquet"),
    "val_data": os.path.join(WORKING_DIR, "val_data.parquet"),
    "test_data": os.path.join(WORKING_DIR, "test_data.parquet"),
    "scaler": os.path.join(WORKING_DIR, "scaler_stats.json"),
    "model": os.path.join(WORKING_DIR, "best_model.pth"),
    "predictions": os.path.join(WORKING_DIR, "test_predictions.npy"),
}

# Final Submission Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Processing Configuration
# ==========================================
RANDOM_STATE = 42
WINDOW_SIZE = 11  # Odd number for symmetric context (+/- 5 epochs)

# Raw columns to load from device_gnss.csv
RAW_GNSS_COLS = [
    "utcTimeMillis",
    "WlsPositionXEcefMeters",
    "WlsPositionYEcefMeters",
    "WlsPositionZEcefMeters",
    "Cn0DbHz",
    "RawPseudorangeUncertaintyMeters",
    "Svid",
]

# Aggregation rules for grouping by epoch
AGG_MAP = {
    "WlsPositionXEcefMeters": "first",
    "WlsPositionYEcefMeters": "first",
    "WlsPositionZEcefMeters": "first",
    "Cn0DbHz": "mean",
    "RawPseudorangeUncertaintyMeters": "mean",
    "Svid": "count",
}

# Renaming map for aggregated columns
AGG_RENAME = {
    "Cn0DbHz": "MeanCn0",
    "RawPseudorangeUncertaintyMeters": "MeanUncertainty",
    "Svid": "SatCount",
}

# Input features for the model (to be standardized)
# Note: 'Delta*' features are derived from Wls* features
INPUT_FEATURES = [
    "WlsLat",
    "WlsLon",
    "WlsAlt",  # Absolute State
    "DeltaLat",
    "DeltaLon",
    "DeltaAlt",  # Dynamics (Velocity)
    "MeanCn0",
    "MeanUncertainty",
    "SatCount",  # Signal Quality
]

# Context features to be injected after the convolutional backbone
# These represent the absolute state of the center timestamp
CONTEXT_FEATURES = ["WlsLat", "WlsLon", "WlsAlt"]

# Targets (Residuals in local metric frame)
TARGETS = ["DeltaEast", "DeltaNorth"]

# ==========================================
# Model & Training Configuration
# ==========================================
MODEL_PARAMS = {
    "input_dim": len(INPUT_FEATURES),
    "context_dim": len(CONTEXT_FEATURES),
    "conv_channels": [32, 64, 128],
    "kernel_size": 3,
    "fc_hidden": [256, 128],
    "dropout": 0.2,
    "output_dim": 2,
}

TRAIN_PARAMS = {
    "batch_size": 512,
    "learning_rate": 1e-3,
    "epochs": 50,
    "patience": 7,  # Early stopping patience
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,
    "num_workers": 4,
    "pin_memory": True,
}

# Debugging / Development
DEBUG = False  # Set to True to use a small subset of data
DEBUG_SAMPLE_SIZE = 1000
