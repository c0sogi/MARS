import os
import torch

# ==========================================
# 1. Directory and File Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_21"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data Paths (using metadata parquets as requested)
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model.pth")
CACHE_DIR = WORKING_DIR  # Directory for caching processed data

# ==========================================
# 2. Hyperparameters & Training Config
# ==========================================
SEED = 42
BATCH_SIZE = 4096
EPOCHS = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 512
NUM_WORKERS = 4
PIN_MEMORY = True

# ==========================================
# 3. Feature Definitions
# ==========================================
ID_COL = "Id"
TARGET_COL = "Cover_Type"

# Raw Continuous Features (10)
RAW_CONTINUOUS_FEATURES = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrol",
    "Vertical_Distance_To_Hydrolog",
    "Horizontal_Distance_To_Roadwa",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_P",
]

# Raw Binary Features (44: 4 Wilderness + 40 Soil Types)
RAW_BINARY_FEATURES = [f"Wilderness_Area{i}" for i in range(1, 5)] + [
    f"Soil_Type{i}" for i in range(1, 41)
]

# Engineered Features (Physics-Informed)
# These names must match the keys created in the preprocessing pipeline
FEAT_ASPECT_SIN = "Aspect_Sin"
FEAT_ASPECT_COS = "Aspect_Cos"
FEAT_EUCLIDEAN_HYDRO = "Euclidean_Distance_To_Hydrology"
FEAT_ABS_HYDRO_ELEV = "Absolute_Hydrology_Elevation"
FEAT_MEAN_AMENITIES = "Mean_Distance_To_Amenities"

ENGINEERED_CONTINUOUS_FEATURES = [
    FEAT_ASPECT_SIN,
    FEAT_ASPECT_COS,
    FEAT_EUCLIDEAN_HYDRO,
    FEAT_ABS_HYDRO_ELEV,
    FEAT_MEAN_AMENITIES,
]

# Final Feature Lists for Model Input
# Strategy: Retain raw Aspect (augmentation), standardize all continuous
FINAL_CONTINUOUS_FEATURES = RAW_CONTINUOUS_FEATURES + ENGINEERED_CONTINUOUS_FEATURES
FINAL_BINARY_FEATURES = RAW_BINARY_FEATURES

# Input Dimensions
NUM_CONTINUOUS = len(FINAL_CONTINUOUS_FEATURES)
NUM_BINARY = len(FINAL_BINARY_FEATURES)
INPUT_DIM = NUM_CONTINUOUS + NUM_BINARY

# ==========================================
# 4. Target Class Mapping
# ==========================================
# Based on analysis: Classes 1, 2, 3, 4, 6, 7 are present. Class 5 is missing.
# We map these to 0-5 for training and map back for submission.
CLASS_LABELS = [1, 2, 3, 4, 6, 7]
NUM_CLASSES = len(CLASS_LABELS)

LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASS_LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(CLASS_LABELS)}


# ==========================================
# 5. Helper Functions
# ==========================================
def get_device():
    """Returns the appropriate torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
