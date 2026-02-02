import os

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# Input directories (Read-Only)
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Output directories (Writable)
# Using 'idea_3' as the working directory for this specific experiment
WORKING_DIR = "./working/idea_3"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
SEED = 42
NUM_WORKERS = 4  # Number of dataloader workers (12 vCPUs available)
DEVICE = "cuda"  # 'cuda' or 'cpu'

# Debugging / Development
# Set DEBUG to True to run on a small subset of data for testing pipelines
DEBUG = False
DEBUG_SAMPLE_SIZE = 100

# =============================================================================
# DATA & SIGNAL PROCESSING CONFIGURATION
# =============================================================================
# Sensor Data Properties
SAMPLING_RATE = 100  # 100 Hz (60001 samples over 10 minutes)
SEGMENT_DURATION = 600  # 10 minutes in seconds
N_SENSORS = 10  # Number of seismic sensors

# Stream A: Tabular Feature Engineering Parameters
SAVGOL_WINDOW = 51  # Window size for smoothing (must be odd)
SAVGOL_POLYORDER = 3  # Polynomial order for smoothing

# Stream B: Spectrogram Generation Parameters
N_FFT = 256  # FFT window size
HOP_LENGTH = 64  # Overlap/Stride
F_MIN = 0  # Minimum frequency
F_MAX = 50  # Maximum frequency (Nyquist limit)
IMG_SIZE = (224, 224)  # Target image size for CNN (H, W)

# =============================================================================
# MODEL CONFIGURATION: STREAM A (LightGBM)
# =============================================================================
# Hyperparameters for the Gradient Boosting Regressor
LGBM_PARAMS = {
    "objective": "regression_l1",  # Mean Absolute Error
    "metric": "mae",
    "n_estimators": 10000,  # High number, controlled by early stopping
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbosity": -1,
    "random_state": SEED,
    "n_jobs": -1,
}

LGBM_EARLY_STOPPING_ROUNDS = 100

# =============================================================================
# MODEL CONFIGURATION: STREAM B (CNN)
# =============================================================================
# Hyperparameters for the Convolutional Neural Network
CNN_PARAMS = {
    "model_name": "resnet18",
    "pretrained": True,
    "in_channels": 10,  # Modified first layer to accept 10 sensor channels
    "dropout": 0.2,
}

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 5  # Patience for Learning Rate Scheduler

# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================
# Weight for Stream A (LightGBM). Stream B (CNN) receives (1 - ENSEMBLE_WEIGHT).
# 0.5 implies equal contribution from both models.
ENSEMBLE_WEIGHT = 0.5
