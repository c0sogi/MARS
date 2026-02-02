import os
import torch

# ==================================================================================
# PATH CONFIGURATION
# ==================================================================================

# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_16"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data File Paths
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Caching Paths (using .parquet and .npy as requested)
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_engineered.parquet")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_engineered.parquet")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_engineered.parquet")

# Scaler Artifacts (saving mean/scale manually to avoid pickle)
SCALER_CENTER_PATH = os.path.join(WORKING_DIR, "scaler_center.npy")
SCALER_SCALE_PATH = os.path.join(WORKING_DIR, "scaler_scale.npy")

# Model and Output Paths
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model.pth")
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==================================================================================
# HYPERPARAMETERS
# ==================================================================================

# General
SEED = 42
DEBUG = False
DEBUG_SAMPLE_SIZE = 1000  # Number of breaths to use if debugging
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Dimensions
SEQ_LEN = 80  # Fixed length of a breath in the dataset

# Model Architecture: Wide-State Weight-Normalized Physics-Injected Composite Network
HIDDEN_DIM = 512  # Width of the LSTM and internal states
NUM_BLOCKS = 4  # Number of Composite Blocks
EXPANSION_FACTOR = (
    2  # Expansion for the Pointwise FFN (2x width) (Cite solution_lesson_node_00052)
)
KERNEL_SIZES = [3, 5, 7]  # For the Multi-Scale CNN Stem
DROPOUT = 0.1  # Applied to residual branch
AUX_WEIGHT = 0.3  # Weight for auxiliary loss (Deep Supervision)

# Optimization
EPOCHS = 20  # Sufficient for convergence with full data
BATCH_SIZE = 512  # Fixed budget
LEARNING_RATE = 1e-3  # Max LR for OneCycle
WEIGHT_DECAY = 1e-2  # AdamW standard
PCT_START = 0.3  # OneCycle warm-up percentage
GRAD_CLIP = 1000.0  # Gradient clipping threshold

# ==================================================================================
# FEATURE ENGINEERING CONFIGURATION
# ==================================================================================

# List of features to be generated and used by the model.
# This ensures consistency between the dataset generation and model input.
FEATURE_NAMES = [
    "time_step",
    "u_in",
    "u_out",
    "R",
    "C",
    "volume",  # Integral of u_in * dt
    "u_in_lag1",  # Lag features
    "u_in_lag2",
    "u_in_lag3",
    "u_in_lag4",
    "u_in_diff1",  # First difference
    "u_in_diff2",  # Second difference
    "u_in_R",  # Interaction: u_in * R
    "vol_C",  # Interaction: volume / C
]

INPUT_DIM = len(FEATURE_NAMES)
