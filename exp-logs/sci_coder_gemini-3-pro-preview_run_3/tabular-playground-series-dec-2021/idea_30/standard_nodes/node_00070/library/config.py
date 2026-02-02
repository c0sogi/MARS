import os
import torch

# =============================================================================
# 1. Paths & Directories
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_30"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Dataset Paths (using metadata parquet files as requested)
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")

# =============================================================================
# 2. Global Settings & Hyperparameters
# =============================================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Debugging / Development
# Set DEBUG to True to run on a smaller subset of data for quick testing
DEBUG = False
DEBUG_SUBSET_SIZE = 10000

# Training Configuration
BATCH_SIZE = 4096
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01  # For AdamW
PATIENCE = 10  # Early Stopping patience

# Scheduler Configuration (ReduceLROnPlateau)
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 5
SCHEDULER_MODE = "max"  # Monitoring accuracy

# Model Architecture
HIDDEN_DIM = 512
DROPOUT = 0.2
NUM_CLASSES = 7  # Target classes are 1-7

# =============================================================================
# 3. Feature Definitions
# =============================================================================
ID_COL = "Id"
TARGET_COL = "Cover_Type"

# Raw Continuous Features (Input)
RAW_CONTINUOUS_FEATURES = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

# Raw Binary Features (Input)
# Soil Types 1-40 and Wilderness Areas 1-4
RAW_BINARY_FEATURES = [
    "Wilderness_Area1",
    "Wilderness_Area2",
    "Wilderness_Area3",
    "Wilderness_Area4",
] + [f"Soil_Type{i}" for i in range(1, 41)]

# Derived Features (To be created in Feature Engineering)
DERIVED_FEATURES = [
    "Aspect_Sin",
    "Aspect_Cos",
    "Hydrology_Distance",  # Euclidean distance
    "Hydrology_Elevation",  # Elevation - Vertical_Distance
    "Mean_Amenities",  # Mean of Hydrology, Roadways, Fire Points
]

# Note: The final continuous input vector will be the concatenation of:
# RAW_CONTINUOUS_FEATURES + DERIVED_FEATURES (Standardized)
# The binary input vector will be RAW_BINARY_FEATURES (0/1)
