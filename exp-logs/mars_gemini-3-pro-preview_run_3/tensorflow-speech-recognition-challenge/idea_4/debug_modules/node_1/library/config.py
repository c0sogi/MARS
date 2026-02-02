import os

# ==========================================
# Reproducibility
# ==========================================
SEED = 42

# ==========================================
# Audio Configuration
# ==========================================
SAMPLE_RATE = 16000
DURATION = 1.0  # seconds
AUDIO_LEN = int(SAMPLE_RATE * DURATION)  # 16000 samples

# Spectrogram Configuration (Multi-Resolution)
# We generate a 3-channel image where each channel corresponds to a different STFT window size.
# This captures features at different time-frequency resolutions.
# Channel 1: 20ms (320 samples) - High Temporal Resolution
# Channel 2: 40ms (640 samples) - Balanced
# Channel 3: 60ms (960 samples) - High Frequency Resolution
WINDOW_SIZES = [320, 640, 960]
N_MELS = 64
HOP_LENGTH = 160  # 10ms hop ensures alignment across all channels (100 time steps)

# ==========================================
# Dataset Configuration
# ==========================================
INPUT_ROOT = "./input"
METADATA_DIR = "./metadata"

TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# Label Definitions
# ==========================================
# The core 10 commands to detect
COMMAND_LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
SILENCE_LABEL = "silence"
UNKNOWN_LABEL = "unknown"

# The full list of 12 classes for the multiclass classification task
ALL_LABELS = COMMAND_LABELS + [SILENCE_LABEL, UNKNOWN_LABEL]
NUM_CLASSES = len(ALL_LABELS)

# Mappings
LABEL2ID = {label: i for i, label in enumerate(ALL_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(ALL_LABELS)}

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 128  # A100 GPU can handle larger batches
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
EARLY_STOPPING_PATIENCE = 5

# ==========================================
# Model Architecture
# ==========================================
# Backbone: ResNet34 (pretrained)
# Neck: Bidirectional GRU
RNN_HIDDEN_SIZE = 128
RNN_LAYERS = 2
DROPOUT = 0.1

# ==========================================
# Augmentation (SpecAugment)
# ==========================================
FREQ_MASK_PARAM = 15
TIME_MASK_PARAM = 20  # Max time mask < 20% of duration (20 steps out of 100)

# ==========================================
# Paths & Directories
# ==========================================
WORK_DIR = "./working/idea_4"
CACHE_DIR = os.path.join(WORK_DIR, "cache")
MODEL_SAVE_PATH = os.path.join(WORK_DIR, "best_model.pth")

SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary directories exist
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
