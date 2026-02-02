import os

# -----------------------------------------------------------------------------
# Global Configuration
# -----------------------------------------------------------------------------
SEED = 42

# -----------------------------------------------------------------------------
# File Paths and Directories
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_8"
SUBMISSION_DIR = "./submission"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data Paths (using Metadata Parquet files)
TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
MODEL_PATH = os.path.join(WORKING_DIR, "swa_model.pth")
CACHE_DIR = WORKING_DIR

# -----------------------------------------------------------------------------
# Data Definition
# -----------------------------------------------------------------------------
ID_COL = "Id"
TARGET_COL = "Cover_Type"

# Raw Feature Lists (as they appear in the dataset)
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

# Raw Binary Features (Wilderness Areas and Soil Types)
RAW_BINARY_FEATURES = [f"Wilderness_Area{i}" for i in range(1, 5)] + [
    f"Soil_Type{i}" for i in range(1, 41)
]

# Final Continuous Features (After Engineering)
# 'Aspect' is replaced by 'Aspect_Sin' and 'Aspect_Cos'
# New geometric features are added
FINAL_CONTINUOUS_FEATURES = [
    "Elevation",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    "Aspect_Sin",
    "Aspect_Cos",
    "Euclidean_Distance_To_Hydrology",
    "Absolute_Hydrology_Elevation",
    "Mean_Distance_To_Amenities",
]

# Binary features are passed through directly
FINAL_BINARY_FEATURES = RAW_BINARY_FEATURES

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
# Parallel DCN-ResNet Architecture
HIDDEN_DIM = 512
NUM_RESNET_BLOCKS = 3
NUM_DCN_LAYERS = 3  # Vector-based DCN v2
DROPOUT_RATE = 0.2
NUM_CLASSES = 7  # Target classes (mapped to 0-6 internally)

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 1024
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4

# Stochastic Weight Averaging (SWA) Settings
SWA_START_EPOCH = 45
SWA_LR = 1e-4
