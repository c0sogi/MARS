import os
import torch

# ==========================================
# 1. Paths and Directories
# ==========================================
INPUT_ROOT = "./input"
METADATA_DIR = "./metadata"

# Audio directories
TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")

# Metadata paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_ROOT, "sample_submission.csv")

# Working directory for checkpoints and cache
WORKING_DIR = "./working/idea_6"
os.makedirs(WORKING_DIR, exist_ok=True)

# Model checkpoint path
MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = "./submission/submission.csv"
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# ==========================================
# 2. Audio Processing Parameters
# ==========================================
SAMPLE_RATE = 16000
DURATION = 1.0  # Seconds
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

# Spectrogram Parameters
# Window size: 25ms -> 16000 * 0.025 = 400
# Hop length: 10ms -> 16000 * 0.010 = 160
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 64
F_MIN = 20
F_MAX = 7800  # Slightly below Nyquist (8000)

# Augmentation Parameters (SpecAugment)
TIME_MASK_PARAM = 20  # Conservative masking (<20% of time steps)
FREQ_MASK_PARAM = 10

# ==========================================
# 3. Model Architecture
# ==========================================
MODEL_NAME = "convnext_tiny"
PRETRAINED = True
IN_CHANNELS = 1  # Spectrogram is 1 channel
NUM_CLASSES = 12

# ==========================================
# 4. Labels and Classes
# ==========================================
# The 10 specific commands + silence + unknown
TARGET_LABELS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "silence",
    "unknown",
]

# Mapping for consistency
LABEL2ID = {label: i for i, label in enumerate(TARGET_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(TARGET_LABELS)}

# ==========================================
# 5. Training Hyperparameters
# ==========================================
SEED = 42
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 4  # 12 vCPUs available, 4 is usually a safe sweet spot
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Scheduler settings
WARMUP_EPOCHS = 2
MIN_LR = 1e-6
