import os
import torch
import random
import numpy as np

# ==========================================
# File Paths and Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_AUDIO_DIR = os.path.join(INPUT_DIR, "train", "audio")
TEST_AUDIO_DIR = os.path.join(INPUT_DIR, "test", "audio")

# Working directory for caching intermediate data (e.g., preprocessed spectrograms)
WORK_DIR = "./working/idea_1"
os.makedirs(WORK_DIR, exist_ok=True)

# Submission directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Audio Processing Parameters
# ==========================================
SAMPLE_RATE = 16000
DURATION = 1.0  # Duration in seconds
AUDIO_LEN = int(SAMPLE_RATE * DURATION)  # 16000 samples

# Spectrogram / Mel-Spectrogram parameters
N_MELS = 40
N_FFT = 480  # Window size (30ms at 16kHz)
HOP_LENGTH = 160  # Hop size (10ms at 16kHz)

# ==========================================
# Label Configuration
# ==========================================
# The 10 specific commands to identify
COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

# The full set of classes for the model output (12 classes)
# Order is important for mapping indices to labels
LABELS = COMMANDS + ["silence", "unknown"]
NUM_CLASSES = len(LABELS)

# Mappings
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
SEED = 42

# Compute device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# Utilities
# ==========================================
def set_seed(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic operations for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
