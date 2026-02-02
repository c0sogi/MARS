import os

# ==========================================
# 1. Paths & Directories
# ==========================================
INPUT_ROOT = "./input"
TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")

METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Working directory for caching and checkpoints
# Using idea_17 as specified in the strategy
WORKING_DIR = "./working/idea_17"
CACHE_DIR = WORKING_DIR
CHECKPOINT_DIR = WORKING_DIR

# Submission
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# 2. Audio Parameters
# ==========================================
SAMPLE_RATE = 16000
DURATION = 1.0  # Seconds
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

# Spectrogram generation (GPU-based)
# Strategy: High-Fidelity Synthesis (Spectral Oversampling)
N_FFT = 1024  # 1024 points for better frequency resolution
HOP_LENGTH = 160  # 10ms at 16kHz (Capture rapid transitions)
WIN_LENGTH = 400  # 25ms at 16kHz
N_MELS = 128  # High resolution Mel bands
F_MIN = 0
F_MAX = 8000  # Nyquist frequency

# ==========================================
# 3. Model Parameters
# ==========================================
# Using EfficientNetV2-B0 for high throughput on GPU
MODEL_NAME = "tf_efficientnetv2_b0"
IN_CHANNELS = 1  # Input is a single-channel spectrogram
DROP_RATE = 0.2
DROP_PATH_RATE = 0.1

# ==========================================
# 4. Labels & Classes
# ==========================================
# The core 10 commands plus silence and unknown
COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
LABELS = COMMANDS + ["silence", "unknown"]
NUM_CLASSES = len(LABELS)

# Mappings
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# ==========================================
# 5. Training Hyperparameters
# ==========================================
SEED = 42
BATCH_SIZE = 32  # Small batch size for high update frequency
EPOCHS = 30  # Max epochs, relies on Early Stopping
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1

# Optimizer / Scheduler
MIN_LR = 1e-6
WARMUP_EPOCHS = 3

# Exponential Moving Average
USE_EMA = True
EMA_DECAY = 0.999

# ==========================================
# 6. Augmentation Parameters
# ==========================================
# SpecAugment (Conservative masking < 20%)
TIME_MASK_PARAM = 20  # Mask up to ~20 time steps (out of ~100)
FREQ_MASK_PARAM = 25  # Mask up to ~25 freq bins (out of 128)
MASK_PROB = 0.5

# Background Noise Mixing
NOISE_PROB = 0.5
NOISE_SNR_MIN = 0  # dB
NOISE_SNR_MAX = 15  # dB

# ==========================================
# 7. Compute & Debug
# ==========================================
NUM_WORKERS = 0  # Data is GPU resident, minimal CPU workers needed
PIN_MEMORY = False  # Not needed if data is already on GPU

# Debugging flags
DEBUG = False
DEBUG_SUBSET_SIZE = 500  # Number of samples to use if DEBUG is True
