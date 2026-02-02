import os
import torch

# -----------------------------------------------------------------------------
# General Configuration
# -----------------------------------------------------------------------------
SEED = 42
NUM_WORKERS = 4  # Number of subprocesses for data loading
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Directory Paths
# -----------------------------------------------------------------------------
INPUT_ROOT = "./input"
TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")
METADATA_DIR = "./metadata"

# Working Directory for Idea 10 (Hybrid 1D-2D Dual-Stream CRNN)
WORKING_DIR = "./working/idea_10"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
MODEL_CHECKPOINT_DIR = WORKING_DIR
BEST_MODEL_PATH = os.path.join(MODEL_CHECKPOINT_DIR, "best_model.pth")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Audio Constants
# -----------------------------------------------------------------------------
SAMPLE_RATE = 16000
DURATION = 1.0  # Seconds
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

# -----------------------------------------------------------------------------
# Feature Extraction (Stream 1: 2D Spectrograms)
# -----------------------------------------------------------------------------
# 3-Channel Multi-Resolution Log-Mel Spectrogram Configuration
# All channels must share the same hop_length to align in time.
# Channel 1: Short window (20ms) -> High temporal resolution
# Channel 2: Medium window (40ms) -> Balanced
# Channel 3: Long window (60ms) -> High frequency resolution
COMMON_HOP_LENGTH = 160  # 10ms -> Resulting width is ~100 frames
N_MELS = 64
F_MIN = 20
F_MAX = 8000

MEL_SPECTROGRAM_CONFIGS = [
    {
        "win_length": int(0.020 * SAMPLE_RATE),  # 320 samples
        "n_fft": 512,
        "hop_length": COMMON_HOP_LENGTH,
        "n_mels": N_MELS,
    },
    {
        "win_length": int(0.040 * SAMPLE_RATE),  # 640 samples
        "n_fft": 1024,
        "hop_length": COMMON_HOP_LENGTH,
        "n_mels": N_MELS,
    },
    {
        "win_length": int(0.060 * SAMPLE_RATE),  # 960 samples
        "n_fft": 2048,
        "hop_length": COMMON_HOP_LENGTH,
        "n_mels": N_MELS,
    },
]

# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------
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
NUM_CLASSES = len(LABELS)
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
# Stream 1 (2D) Backbone (SK-ResNet34)
RESNET_LAYERS = [3, 4, 6, 3]
# Adjusted strides to preserve time dimension in later stages for RNN processing
RESNET_STRIDES = [(1, 1), (2, 2), (2, 2), (1, 1)]

# Fusion & Sequence Modeling
RNN_HIDDEN_DIM = 256
RNN_NUM_LAYERS = 2
RNN_DROPOUT = 0.3
BIDIRECTIONAL = True

# Attention Head
ATTN_NUM_HEADS = 4
ATTN_HIDDEN_DIM = 128

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

# Augmentation
SPEC_AUG_TIME_MASK_PARAM = 20
SPEC_AUG_FREQ_MASK_PARAM = 10
SPEC_AUG_TIME_MASK_LIMIT = 0.2  # Max 20% of duration
