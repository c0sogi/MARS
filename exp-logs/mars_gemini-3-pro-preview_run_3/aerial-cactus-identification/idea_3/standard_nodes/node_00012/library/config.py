import os
import torch

# =============================================================================
# PATHS AND DIRECTORIES
# =============================================================================

# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_3")
SUBMISSION_DIR = "./submission"

# Ensure working/output directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata paths (pre-generated)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Image directories
# Metadata contains relative paths (e.g., 'train/id.jpg'), so we rely on INPUT_DIR
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

# Output paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
BEST_MODEL_PATH = os.path.join(CACHE_DIR, "best_model.pth")

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

IMAGE_SIZE = (32, 32)
NUM_CLASSES = 1  # Binary classification
INPUT_CHANNELS = 3

# Normalization constants (Standard 0.5 mean/std for robust initialization)
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

SEED = 42
BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Early Stopping
EARLY_STOPPING_PATIENCE = 20  # Extended patience for Mixup
EARLY_STOPPING_MODE = "max"  # Monitor AUC (maximize)

# Regularization
MIXUP_ALPHA = 1.0

# Scheduler (Cosine Annealing)
T_MAX = NUM_EPOCHS
ETA_MIN = 1e-6

# =============================================================================
# MODEL ARCHITECTURE CONFIGURATIONS
# =============================================================================

MODEL_CONFIGS = {
    "wide_se_resnet": {
        "name": "WideSEResNet",
        "depth": 28,
        "widen_factor": 10,
        "drop_rate": 0.3,
        "stem_type": "cifar",  # 3x3 conv, stride 1
        "se_reduction": 16,
    },
    "densenet_bc": {
        "name": "DenseNetBC",
        "growth_rate": 12,
        "block_config": (16, 16, 16),
        "compression": 0.5,
        "num_init_features": 24,
        "stem_type": "cifar",  # 3x3 conv, stride 1
        "drop_rate": 0.2,
    },
}

# =============================================================================
# COMPUTE SETTINGS
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Optimized for 12 vCPUs

# =============================================================================
# DEBUGGING
# =============================================================================

# Set to integer (e.g., 100) to limit dataset size for fast debugging
# Set to None for full training run
DEBUG_SAMPLE_SIZE = None
