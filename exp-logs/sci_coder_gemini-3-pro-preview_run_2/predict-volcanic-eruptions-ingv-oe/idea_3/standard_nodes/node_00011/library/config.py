import os

# -------------------------------------------------------------------
# Project Paths
# -------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_3"
SUBMISSION_DIR = "./submission"

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Artifact Storage
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model.pth")
SCALER_MEAN_PATH = os.path.join(WORKING_DIR, "scaler_mean.npy")
SCALER_SCALE_PATH = os.path.join(WORKING_DIR, "scaler_scale.npy")
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -------------------------------------------------------------------
# Data Parameters
# -------------------------------------------------------------------
NUM_SENSORS = 10
SAMPLING_RATE = 100  # Hz (Approx 60001 samples over 10 mins)
SIGNAL_LENGTH = 60001
DURATION_SECONDS = 600

# -------------------------------------------------------------------
# Feature Engineering (Spectrogram) Parameters
# -------------------------------------------------------------------
# Parameters tuned for seismic signals and EfficientNet input
N_FFT = 1024
HOP_LENGTH = 512  # Results in ~118 time steps (60001 / 512)
N_MELS = 128  # Frequency bins
FMIN = 0
FMAX = 50  # Nyquist frequency for 100Hz sampling

# -------------------------------------------------------------------
# Model Architecture Hyperparameters
# -------------------------------------------------------------------
BACKBONE_NAME = "efficientnet_b0"
PRETRAINED = True
# RNN Head
RNN_HIDDEN_DIM = 256
RNN_NUM_LAYERS = 2
RNN_BIDIRECTIONAL = True
RNN_DROPOUT = 0.3
# MLP Head (for stats)
MLP_HIDDEN_DIM = 256
DROPOUT_RATE = 0.3

# -------------------------------------------------------------------
# Training Hyperparameters
# -------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
EPOCHS = 50  # Extended training duration for convergence
PATIENCE = 15  # High patience for noisy seismic data validation
NUM_WORKERS = 4
DEVICE = "cuda"  # Default to CUDA, handled in training script


# -------------------------------------------------------------------
# Setup Utilities
# -------------------------------------------------------------------
def setup_directories():
    """
    Creates the working and submission directories if they do not exist.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
