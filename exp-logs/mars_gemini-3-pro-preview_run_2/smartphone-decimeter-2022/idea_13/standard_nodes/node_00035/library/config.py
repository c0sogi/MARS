import os

# ==========================================
# File Paths and Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_13"
SUBMISSION_DIR = "./submission"

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Sample Submission Path
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Data Processing Hyperparameters
# ==========================================
WINDOW_SIZE = 15  # Size of the sliding window (number of epochs)
RANDOM_STATE = 42  # Fixed seed for reproducibility
DEBUG = False  # Set to True to run on a small subset of data for debugging

# ==========================================
# Model Architecture Hyperparameters
# ==========================================
# Multi-Scale CNN Backbone
KERNEL_SIZES = [3, 5, 7]  # Kernel sizes for parallel 1D convolution branches
CNN_CHANNELS = 64  # Number of output channels for each CNN branch
CNN_DROPOUT = 0.2  # Dropout rate within CNN blocks

# Environmental Context & Fusion Head
CONTEXT_EMBED_DIM = 32  # Dimension of the environmental context embedding
FUSION_HIDDEN_DIM = 128  # Hidden dimension of the fusion MLP
FUSION_DROPOUT = 0.2  # Dropout rate in the fusion MLP

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
PATIENCE = 5  # Early stopping patience (epochs without improvement)
WEIGHT_DECAY = 1e-4  # Weight decay for AdamW optimizer
SCHEDULER_FACTOR = 0.5  # Factor for ReduceLROnPlateau
SCHEDULER_PATIENCE = 2  # Patience for ReduceLROnPlateau

# ==========================================
# Feature Definitions
# ==========================================
# Features used in the Kinematic Stream (Time-Series Input)
# These are computed per timestep within the window
KINEMATIC_FEATURES = [
    "rel_lat_m",  # Latitude relative to window center (meters)
    "rel_lon_m",  # Longitude relative to window center (meters)
    "rel_alt_m",  # Altitude relative to window center (meters)
    "d_lat_m",  # Delta Latitude (velocity proxy)
    "d_lon_m",  # Delta Longitude (velocity proxy)
    "d_alt_m",  # Delta Altitude (velocity proxy)
    "Cn0DbHz",  # Carrier-to-noise density (standardized)
    "Uncertainty",  # RawPseudorangeUncertaintyMeters (standardized)
]

# Features used in the Environmental Context Stream (Vector Input)
# These are aggregated statistics over the entire window
CONTEXT_FEATURES = [
    "mean_sv_elevation",  # Mean satellite elevation (proxy for sky visibility)
    "std_sv_elevation",  # Std dev of satellite elevation
    "mean_cn0",  # Mean signal strength
    "mean_uncertainty",  # Mean uncertainty
]

# Target Variables (Residuals in Local Metric Frame)
TARGET_COLS = ["res_east_m", "res_north_m"]
