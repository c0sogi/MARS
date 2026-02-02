import os

# =============================================================================
# PATHS AND DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
TRAIN_AUDIO_DIR = os.path.join(INPUT_DIR, "train", "audio")
TEST_AUDIO_DIR = os.path.join(INPUT_DIR, "test", "audio")
BACKGROUND_NOISE_DIR = os.path.join(TRAIN_AUDIO_DIR, "_background_noise_")

METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working directory for Idea 11 (checkpoints, cache, submissions)
WORKING_DIR = "./working/idea_11"
os.makedirs(WORKING_DIR, exist_ok=True)

# =============================================================================
# AUDIO PROCESSING PARAMETERS
# =============================================================================
SAMPLE_RATE = 16000
DURATION = 1.0  # Duration of clips in seconds
N_FFT = 1024  # Spectral Oversampling (1024 vs default 400)
WIN_LENGTH = 400  # 25ms window size (16000 * 0.025)
HOP_LENGTH = 160  # 10ms hop length (16000 * 0.010)
N_MELS = 64  # Number of Mel bands

# =============================================================================
# MODEL ARCHITECTURE PARAMETERS
# =============================================================================
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 12
PRETRAINED = True
USE_ATTENTION_POOLING = True
IN_CHANNELS = 3  # 3 Channels: Spectrogram, Delta, Delta-Delta

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEED = 42
BATCH_SIZE = 32  # Small batch size for frequent updates
EPOCHS = 20
LEARNING_RATE = 1e-3
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 1e-4  # Standard regularization for AdamW
NUM_WORKERS = 4  # Number of dataloader workers (safe for 12 vCPUs)

# =============================================================================
# LABELS AND MAPPINGS
# =============================================================================
# The 10 specific commands to identify
TARGET_COMMANDS = [
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
]

# Full label set including auxiliary classes
# Order is critical: 10 commands -> silence -> unknown
LABELS = TARGET_COMMANDS + ["silence", "unknown"]

# Mappings
LABEL2INT = {label: idx for idx, label in enumerate(LABELS)}
INT2LABEL = {idx: label for idx, label in enumerate(LABELS)}

# =============================================================================
# DEBUGGING AND DEVELOPMENT
# =============================================================================
# Flags to control dataset size for rapid prototyping
DEBUG = False
DEBUG_SAMPLE_SIZE = 100
