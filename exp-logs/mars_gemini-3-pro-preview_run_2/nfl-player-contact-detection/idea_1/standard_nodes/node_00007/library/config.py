import os
import torch

# =============================================================================
# Directories and Paths
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_1"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Raw Data Paths
TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Cache Paths for Processed Features
# Using Parquet for efficient storage of tabular data
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_features_v3.parquet")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_features_v3.parquet")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_features_v3.parquet")

# Model Checkpoint Path
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "k_mlp_model.pth")
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Global Settings
# =============================================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Number of workers for data loading

# =============================================================================
# Data Processing Hyperparameters
# =============================================================================
# Temporal window size: +/- 5 steps around the target step (Total 11 steps)
WINDOW_SIZE = 11
HALF_WINDOW = (WINDOW_SIZE - 1) // 2

# Columns to extract from tracking data for feature engineering
RAW_TRACKING_COLS = [
    "x_position",
    "y_position",
    "speed",
    "acceleration",
    "orientation",
    "direction",
    "sa",  # Signed acceleration
]

# =============================================================================
# Model Architecture (K-MLP)
# =============================================================================
# List of neuron counts for hidden layers
HIDDEN_LAYERS = [512, 256, 128, 64]
DROPOUT_RATE = 0.3

# =============================================================================
# Training Hyperparameters
# =============================================================================
BATCH_SIZE = 2048
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 15
PATIENCE = 5  # Early stopping patience

# Class Imbalance Handling
# Using Focal Loss to address imbalance without severe probability shift
# Alpha: Balance factor (0.75 favors positive class slightly to counter rarity)
# Gamma: Focusing parameter (2.0 is standard)
FOCAL_LOSS_ALPHA = 0.75
FOCAL_LOSS_GAMMA = 2.0
