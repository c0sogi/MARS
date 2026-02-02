import os
import torch

# =============================================================================
# Directories and File Paths
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_21_retry"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data Paths
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")
SCALER_PATH = os.path.join(WORKING_DIR, "scaler.pkl")

# =============================================================================
# Global Configuration
# =============================================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

# =============================================================================
# Feature Engineering Configuration
# =============================================================================
# Raw Columns
ID_COL = "id"
BREATH_ID_COL = "breath_id"
TIME_COL = "time_step"
TARGET_COL = "pressure"
U_IN_COL = "u_in"
U_OUT_COL = "u_out"
R_COL = "R"
C_COL = "C"

# Engineering Flags
USE_LAGS = True
LAG_STEPS = [1, 2, 3, 4]
USE_DIFFS = True
USE_INTEGRATION = True  # Calculates volume from u_in * dt
USE_INTERACTIONS = True  # Creates R*u_in, volume/C

# =============================================================================
# Model Architecture: Graduated-Capacity Curated-Composite Network
# =============================================================================
# Stem (Bottleneck Initialization)
STEM_DIM = 512
KERNEL_SIZES = [3, 5, 7]  # Multi-scale 1D Convolution kernels

# Backbone (Wide-State Expansion)
WIDE_DIM = 1024
LSTM_HIDDEN = 512  # Bidirectional: 512 * 2 = 1024 output matches WIDE_DIM
DROPOUT = 0.1

# Heads
AUX_WEIGHT = 0.3  # Weight for the auxiliary regression head

# =============================================================================
# Training Hyperparameters
# =============================================================================
EPOCHS = 35
BATCH_SIZE = 512

# Optimizer (AdamW + OneCycleLR)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2

# Stability
CLIP_GRAD = 1.0  # Strict clipping for wide-state stability

# Early Stopping
EARLY_STOPPING_PATIENCE = 7


def get_config_dict():
    """
    Returns a dictionary of the configuration for logging or hashing purposes.
    """
    return {
        "SEED": SEED,
        "STEM_DIM": STEM_DIM,
        "WIDE_DIM": WIDE_DIM,
        "LSTM_HIDDEN": LSTM_HIDDEN,
        "KERNEL_SIZES": KERNEL_SIZES,
        "DROPOUT": DROPOUT,
        "AUX_WEIGHT": AUX_WEIGHT,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "CLIP_GRAD": CLIP_GRAD,
        "USE_LAGS": USE_LAGS,
        "USE_DIFFS": USE_DIFFS,
        "USE_INTEGRATION": USE_INTEGRATION,
        "USE_INTERACTIONS": USE_INTERACTIONS,
        "CONTEXT_FEATURES": CONTEXT_FEATURES,
    }
