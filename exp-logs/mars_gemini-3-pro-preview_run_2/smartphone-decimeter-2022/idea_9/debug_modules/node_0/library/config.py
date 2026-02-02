import os

# ==========================================
# File Paths and Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_9"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# ==========================================
# Data Processing & Feature Engineering
# ==========================================
RANDOM_STATE = 42
WINDOW_SIZE = 15  # Size of the sliding window (number of epochs)

# Approximate conversion factors
DEG_TO_M_LAT = 111319.9
DEG_TO_M_LON = 111319.9  # Will be adjusted by cos(lat) in preprocessing if needed, or treated as approx

# Features for the Trajectory Stream (Time-Series Input: [Batch, Window, Features])
# These capture the motion dynamics and signal quality over time.
TRAJECTORY_FEATURES = [
    "rel_wls_lat_m",  # WLS Latitude relative to window center (meters)
    "rel_wls_lon_m",  # WLS Longitude relative to window center (meters)
    "vel_lat_m",  # First-order difference of WLS Latitude (meters)
    "vel_lon_m",  # First-order difference of WLS Longitude (meters)
    "vel_alt_m",  # First-order difference of WLS Altitude (meters)
    "mean_cn0",  # Mean Carrier-to-Noise density for the epoch
    "mean_uncertainty",  # Mean Raw Pseudorange Uncertainty for the epoch
]

# Features for the Context Stream (Static Input per Window: [Batch, Features])
# These capture the satellite geometry and environmental conditions.
CONTEXT_FEATURES = [
    "sv_count",  # Number of satellites visible
    "mean_sv_elevation",  # Mean elevation of visible satellites
    "std_sv_elevation",  # Standard deviation of satellite elevation
    "min_sv_elevation",  # Minimum satellite elevation (indicates obstruction)
    "mean_sv_azimuth",  # Mean azimuth
    "std_sv_azimuth",  # Standard deviation of azimuth (sky spread)
]

# Target Variables
# We predict the residual error in meters to add to the WLS baseline.
TARGET_FEATURES = [
    "res_lat_m",  # Ground Truth Lat - WLS Lat (in meters)
    "res_lon_m",  # Ground Truth Lon - WLS Lon (in meters)
]

# Metadata columns to keep for identification and submission
META_FEATURES = [
    "tripId",
    "UnixTimeMillis",
    "drive_id",
    "phone_name",
    "WlsPositionXEcefMeters",  # Kept for reconstruction if needed
    "WlsPositionYEcefMeters",
    "WlsPositionZEcefMeters",
]

# ==========================================
# Model Hyperparameters
# ==========================================
# Dimensions
TRAJECTORY_INPUT_DIM = len(TRAJECTORY_FEATURES)
CONTEXT_INPUT_DIM = len(CONTEXT_FEATURES)
HIDDEN_DIM = 128
OUTPUT_DIM = 2  # Latitude and Longitude residuals

# Architecture
CNN_LAYERS = 3
KERNEL_SIZE = 3
DROPOUT_RATE = 0.2

# ==========================================
# Training Settings
# ==========================================
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
NUM_WORKERS = 4

# Debugging
DEBUG = False  # Set to True to use a smaller subset of data
DEBUG_SAMPLE_SIZE = 1000  # Number of trips/samples to use in debug mode
