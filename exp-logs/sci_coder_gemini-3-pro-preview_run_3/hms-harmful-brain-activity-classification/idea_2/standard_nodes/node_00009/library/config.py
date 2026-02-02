import os
import torch

# ==========================================
# Directories & Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
IDEA_DIR = os.path.join(WORKING_DIR, "idea_2")
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(IDEA_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Data Paths
TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")

# Output Paths
MODEL_PATH = os.path.join(IDEA_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
CHECKPOINT_PATH = os.path.join(IDEA_DIR, "checkpoint.pth")

# ==========================================
# Signal Processing Parameters
# ==========================================
SAMPLING_RATE = 200  # Hz
DURATION = 50  # Seconds
TOTAL_SAMPLES = SAMPLING_RATE * DURATION  # 10,000 samples
DOWNSAMPLE_FACTOR = 4  # Downsample to 50Hz

# Spectrogram Generation
# N_FFT=1024 (approx 5s) is too large for 50s clips if we want fine temporal resolution.
# Using N_FFT=256 (1.28s) and HOP_LENGTH=64 (0.32s) gives ~150 time steps.
N_FFT = 256
HOP_LENGTH = 64
N_MELS = 128
FMIN = 0.5  # High-pass filter equivalent
FMAX = 100.0  # Nyquist limit

# Standard 10-20 System EEG Channels
# We select these specific channels to form a standard montage
EEG_CHANNELS = [
    "Fp1",
    "F3",
    "C3",
    "P3",
    "F7",
    "T3",
    "T5",
    "O1",
    "Fz",
    "Cz",
    "Pz",
    "Fp2",
    "F4",
    "C4",
    "P4",
    "F8",
    "T4",
    "T6",
    "O2",
]
N_CHANNELS = len(EEG_CHANNELS)

# ==========================================
# Model & Input Parameters
# ==========================================
# The model will receive a stacked image of (N_CHANNELS * N_MELS, TIME_STEPS)
# We resize this to a square for EfficientNet
IMG_SIZE = (512, 512)  # (Height, Width)
IN_CHANNELS = 3  # EfficientNet expects 3 channels (we can replicate grayscale)

# ==========================================
# Training Hyperparameters
# ==========================================
SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 3  # Early stopping patience

# Hardware
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Debugging / Development
DEBUG = False  # Set to True to train on a small subset
DEBUG_SIZE = 1000  # Number of samples to use in debug mode

# ==========================================
# Targets & Labels
# ==========================================
TARGET_COLS = [
    "seizure_prob",
    "lpd_prob",
    "gpd_prob",
    "lrda_prob",
    "grda_prob",
    "other_prob",
]
CLASS_NAMES = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
NUM_CLASSES = len(TARGET_COLS)
