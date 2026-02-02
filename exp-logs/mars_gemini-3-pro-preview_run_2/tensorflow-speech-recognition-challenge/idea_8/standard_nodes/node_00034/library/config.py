import os
import torch
import numpy as np
import random

# ==========================================
# 1. Paths and Directories
# ==========================================
INPUT_ROOT = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_8"

# Ensure working directory exists for caching and outputs
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Audio Source Paths
TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")

# ==========================================
# 2. Audio Processing Parameters
# ==========================================
SAMPLE_RATE = 16000
DURATION = 1.0  # seconds
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

# Spectrogram Parameters (High-Fidelity)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 160  # 10ms at 16kHz
WIN_LENGTH = 400  # 25ms at 16kHz
F_MIN = 20
F_MAX = 8000  # Nyquist frequency

# ==========================================
# 3. Label Configuration
# ==========================================
# The 10 specific commands to identify
TARGET_LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

# Complete list of classes for the classifier (12 classes)
# Structure: [Auxiliary Labels] + [Sorted Target Labels]
LABELS = ["silence", "unknown"] + sorted(TARGET_LABELS)
NUM_CLASSES = len(LABELS)

# Mappings for easy lookup
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}

# ==========================================
# 4. Training Hyperparameters
# ==========================================
SEED = 42
BATCH_SIZE = (
    32  # Smaller batch size for more gradient updates (Cite solution_lesson_node_00031)
)
NUM_EPOCHS = 50  # Extended training for convergence
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1  # Regularization
NUM_WORKERS = 4  # Data loading workers
PATIENCE = 10  # Increased patience

# Compute Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 5. Utilities
# ==========================================
def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
