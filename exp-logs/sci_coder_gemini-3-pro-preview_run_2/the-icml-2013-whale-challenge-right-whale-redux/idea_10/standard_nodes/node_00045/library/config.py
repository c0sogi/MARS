import os
import torch

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================
INPUT_ROOT = "./input"
TRAIN_DIR = os.path.join(INPUT_ROOT, "train2")
TEST_DIR = os.path.join(INPUT_ROOT, "test2")

# Metadata paths
METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Output directories
WORK_DIR = "./working/idea_10"
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure working directories exist
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# AUDIO PREPROCESSING PARAMETERS
# =============================================================================
SR = 2000  # Sample rate (derived from data analysis)
DURATION = 2.0  # Approximate duration in seconds
N_FFT = 1024  # High frequency resolution
HOP_LENGTH = 64  # High temporal resolution
N_MELS = 128  # Number of Mel bands
FMIN = 0
FMAX = None
NORMALIZED = False  # Disable area normalization to preserve pink noise tilt

# =============================================================================
# MODEL PARAMETERS
# =============================================================================
ARCH = "efficientnet_b0"
PRETRAINED = True
NUM_CLASSES = 1
IN_CHANNELS = 1
POOLING = "gem"  # Generalized Mean Pooling

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEED = 42
N_FOLDS = 5
BATCH_SIZE = 128
NUM_EPOCHS = 50  # Max epochs, controlled by Early Stopping
PATIENCE = 6  # Early stopping patience
LR = 1e-3  # Initial learning rate
WEIGHT_DECAY = 1e-4  # Weight decay for AdamW
NUM_WORKERS = 4  # Number of DataLoader workers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# DEBUGGING / DEVELOPMENT
# =============================================================================
DEBUG = False  # Set to True to run on a small subset for testing
DEBUG_SAMPLES = 100  # Number of samples to use in debug mode
