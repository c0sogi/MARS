import os

# ==========================================
# Path Configuration
# ==========================================
INPUT_ROOT = "./input"
TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")
BACKGROUND_NOISE_DIR = os.path.join(TRAIN_AUDIO_DIR, "_background_noise_")

METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sample_submission.csv")

# Directory for caching and saving models/predictions
WORKING_DIR = "./working/idea_13"
os.makedirs(WORKING_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# ==========================================
# Audio Parameters
# ==========================================
SAMPLE_RATE = 16000
DURATION = 1.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)

# ==========================================
# Feature Extraction Parameters
# ==========================================
N_MELS = 64

# Multi-resolution settings
# These values correspond to 20ms (320), 40ms (640), and 60ms (960) at 16kHz.
# Used for the 3-channel spectrogram generation in the 2D stream.
HOP_LENGTHS = [320, 640, 960]
WIN_LENGTHS = [320, 640, 960]
F_MIN = 20
F_MAX = 8000

# ==========================================
# Model Architecture Parameters
# ==========================================
N_CLASSES = 12
HIDDEN_DIM = 256
DROPOUT = 0.5

# ==========================================
# Training Parameters
# ==========================================
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 20
SEED = 42
NUM_WORKERS = 4  # Adjust based on available vCPUs (12 available)

# ==========================================
# Label Definition
# ==========================================
# The 10 specific commands + silence + unknown
LABELS = [
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

# Mappings
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}
