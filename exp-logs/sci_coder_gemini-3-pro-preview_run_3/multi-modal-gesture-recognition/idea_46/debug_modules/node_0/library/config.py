import os

# ==========================================
# Global Configuration
# ==========================================

# Reproducibility
SEED = 42


# ==========================================
# File Paths & Directories
# ==========================================
class Paths:
    INPUT = "./input"
    METADATA = "./metadata"

    # Working directory for this specific idea
    WORKING_DIR = "./working/idea_46"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    PREDICTION_DIR = os.path.join(WORKING_DIR, "predictions")
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA, "train.csv")
    VAL_CSV = os.path.join(METADATA, "val.csv")
    TEST_CSV = os.path.join(METADATA, "test.csv")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")


# Ensure working directories exist
os.makedirs(Paths.CACHE_DIR, exist_ok=True)
os.makedirs(Paths.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Paths.PREDICTION_DIR, exist_ok=True)
os.makedirs(Paths.SUBMISSION_DIR, exist_ok=True)


# ==========================================
# Data Processing Configuration
# ==========================================
class DataConfig:
    # Sampling
    WINDOW_SIZE = 64
    STRIDE = 32

    # Skeleton Features
    USE_RAW_MM = True  # Use millimeters (~1000 scale) instead of meters
    CENTER_SKELETON = True  # Root-relative centering
    USE_CENTRAL_DIFFERENCE = True  # Use np.gradient instead of np.diff
    NUM_JOINTS = 20

    # Audio Features
    AUDIO_SR = 16000
    N_MFCC = 13

    # Classes
    # 20 Gestures + 1 Background (Class 0)
    NUM_CLASSES = 21

    # Debugging / Development
    # Set to None to use full dataset, or an integer (e.g., 100) for quick testing
    DEBUG_SAMPLE_SIZE = None


# ==========================================
# Model Architecture Configuration
# ==========================================
class ModelConfig:
    # Stage 1: Bi-GRU Encoder
    GRU_LAYERS = 2
    GRU_HIDDEN_SIZE = 96  # Per direction (Total 192)
    GRU_DROPOUT = 0.4

    # Stage 2 & 3: TCN Refinement
    # Monotonically increasing dilation for RF=63 (matches window 64)
    TCN_DILATIONS = [1, 2, 4, 8, 16]
    TCN_KERNEL_SIZE = 3
    TCN_CHANNELS = 64
    TCN_DROPOUT = 0.2


# ==========================================
# Training Configuration
# ==========================================
class TrainConfig:
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Loss Weights
    # Weight for background class (index 0) to handle imbalance
    BACKGROUND_WEIGHT = 0.2

    # Deep Supervision Weights (Stage 1, Stage 2, Stage 3)
    LOSS_STAGES_WEIGHTS = [1.0, 1.0, 1.0]

    # Log-Space Smoothing Loss
    SMOOTHING_LAMBDA = 0.15
    SMOOTHING_THRESHOLD = 1.0

    # Early Stopping
    PATIENCE = 10


# ==========================================
# Label Mapping
# ==========================================
# Maps gesture names to IDs (1-20).
# Class 0 is reserved for 'background' (no gesture).
LABEL_MAP = {
    "vattene": 1,
    "vieniqui": 2,
    "perfetto": 3,
    "furbo": 4,
    "cheduepalle": 5,
    "chevuoi": 6,
    "daccordo": 7,
    "seipazzo": 8,
    "combinato": 9,
    "freganiente": 10,
    "ok": 11,
    "cosatifarei": 12,
    "basta": 13,
    "prendere": 14,
    "noncenepiu": 15,
    "fame": 16,
    "tantotempo": 17,
    "buonissimo": 18,
    "messidaccordo": 19,
    "sonostufo": 20,
}

# Reverse mapping for decoding
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
