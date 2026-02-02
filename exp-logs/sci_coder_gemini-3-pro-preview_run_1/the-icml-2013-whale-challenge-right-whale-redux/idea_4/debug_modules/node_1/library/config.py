import os
import torch

# ==========================================
# File Paths & Directories
# ==========================================
INPUT_ROOT = "./input"
METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Working directory for caching processed data and saving models
WORKING_DIR = "./working/idea_4"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Audio Processing Parameters
# ==========================================
SAMPLE_RATE = 2000

# Spectrogram extraction
# Window size: 25ms. At 2000Hz, 0.025 * 2000 = 50 samples.
WIN_LENGTH = 50
# Hop length: 10ms. At 2000Hz, 0.010 * 2000 = 20 samples.
HOP_LENGTH = 20
# N_FFT: Size of FFT. Must be >= WIN_LENGTH.
# Using 512 to ensure sufficient frequency resolution for 128 Mel bins.
N_FFT = 512
N_MELS = 128

# Frequency limits
F_MIN = 20
F_MAX = 1000  # Nyquist frequency for 2kHz sample rate

# ==========================================
# Augmentation Parameters
# ==========================================
# Mixup
MIXUP_ALPHA = 0.4

# SpecAugment
# Time Mask: Max 200ms. With 10ms hop, 200ms = 20 frames.
TIME_MASK_PARAM = 20
FREQ_MASK_PARAM = 20

# ==========================================
# Model Architecture Parameters
# ==========================================
MODEL_NAME = "efficientnet_b0"
HIDDEN_DIM = 128  # Dimension for the BiGRU and Attention layers
NUM_CLASSES = 1  # Binary classification

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 20
SEED = 42

# Compute
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Debugging / Development
# ==========================================
# Set DEBUG to True to run on a small subset of data for quick testing
DEBUG = False
DEBUG_SUBSET_SIZE = 100
