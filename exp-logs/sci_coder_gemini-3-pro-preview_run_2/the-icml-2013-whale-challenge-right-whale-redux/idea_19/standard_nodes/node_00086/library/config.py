import os

# =============================================================================
# DIRECTORY AND PATH CONFIGURATION
# =============================================================================
INPUT_ROOT = "./input"
TRAIN_DIR = os.path.join(INPUT_ROOT, "train2")
TEST_DIR = os.path.join(INPUT_ROOT, "test2")

METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Main working directory for this specific experiment (Idea 19)
WORK_DIR = "./working/idea_19"
os.makedirs(WORK_DIR, exist_ok=True)

# Sub-directories for checkpoints and cache
CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CACHE_DIR = os.path.join(WORK_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Submission directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# AUDIO PROCESSING CONFIGURATION
# =============================================================================
SAMPLE_RATE = 2000
N_FFT = 1024  # High Frequency Resolution
HOP_LENGTH = 64  # High Time Resolution
N_MELS = 128
FMIN = 0
FMAX = None  # Defaults to Nyquist (SR/2)
TOP_DB = 80  # Dynamic range clamping

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Heterogeneous ensemble architectures
MODEL_ARCHITECTURES = ["tf_efficientnet_b0_ns", "resnet34"]
NUM_CLASSES = 1
USE_GEM_POOLING = True
PRETRAINED = True

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEED = 42
N_FOLDS = 5
BATCH_SIZE = 128
NUM_EPOCHS = 20

# Optimization
LR = 1e-3
WEIGHT_DECAY = 1e-4  # Low weight decay for Noisy Student weights

# Augmentation (SpecAugment)
FREQ_MASK_PARAM = 26  # Aggressive frequency masking (Cite solution_lesson_node_00071)
TIME_MASK_PARAM = 10

# Hardware
NUM_WORKERS = 4

# =============================================================================
# ENSEMBLE CONFIGURATION
# =============================================================================
# Multi-Objective Checkpointing
SAVE_METRICS = ["auc", "loss"]  # Save separate checkpoints for best AUC and best Loss
