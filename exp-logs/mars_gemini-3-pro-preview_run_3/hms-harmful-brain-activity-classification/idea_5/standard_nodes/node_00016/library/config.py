import os
import torch

# =============================================================================
# DIRECTORIES & PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Files
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Data Directories
TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
TEST_SPECS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

# Model Checkpoint
MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# =============================================================================
# DATA SPECIFICATIONS
# =============================================================================
# Labels
TARGET_COLS = [
    "seizure_prob",
    "lpd_prob",
    "gpd_prob",
    "lrda_prob",
    "grda_prob",
    "other_prob",
]
CLASS_NAMES = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
NUM_CLASSES = len(CLASS_NAMES)

# EEG Parameters (Stream A)
EEG_RAW_SAMPLE_RATE = 200  # Hz
EEG_TARGET_SAMPLE_RATE = 50  # Hz (Downsampling)
EEG_DURATION_SEC = 50
EEG_SEQ_LENGTH = int(EEG_DURATION_SEC * EEG_TARGET_SAMPLE_RATE)  # 2500
EEG_CHANNELS_COUNT = 19

# Standard 10-20 System Channels (excluding EKG)
EEG_CHANNELS = [
    "Fp1",
    "F3",
    "C3",
    "P3",
    "F7",
    "T3",
    "T5",
    "O1",  # Left
    "Fz",
    "Cz",
    "Pz",  # Center
    "Fp2",
    "F4",
    "C4",
    "P4",
    "F8",
    "T4",
    "T6",
    "O2",  # Right
]

# Spectrogram Parameters (Stream B)
SPEC_DURATION_SEC = 600  # 10 minutes
SPEC_HEIGHT = 256
SPEC_WIDTH = 256
SPEC_CHANNELS = 4  # LL, RL, LP, RP (stacked as depth)

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
MAX_GRAD_NORM = 1.0

# Early Stopping
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.001

# Hardware
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# MODEL ARCHITECTURE CONFIG
# =============================================================================
# Stream A (EEG)
RESNET_BLOCKS = [2, 2, 2, 2]  # ResNet18-like structure for 1D
RESNET_FILTERS = [64, 128, 256, 512]
EEG_DROPOUT = 0.2

# Stream B (Spectrogram)
EFFICIENTNET_VERSION = "efficientnet_b0"
SPEC_DROPOUT = 0.2

# Fusion
FUSION_HIDDEN_DIM = 256
FUSION_DROPOUT = 0.3
