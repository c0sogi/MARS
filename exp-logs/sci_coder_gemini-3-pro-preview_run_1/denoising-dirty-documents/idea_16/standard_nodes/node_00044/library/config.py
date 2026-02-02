import os
import torch
import random
import numpy as np

# =============================================================================
# 1. PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_16"
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure mutable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# =============================================================================
# 2. HARDWARE & REPRODUCIBILITY
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Optimized for the available 12 vCPUs


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# 3. DATA PIPELINE CONFIGURATION
# =============================================================================
IMG_CHANNELS = 1

# Stream A: Context Specialists
# Trained on larger crops to capture global structure
STREAM_A_CONFIG = {
    "name": "StreamA_Context",
    "img_size": (320, 320),
    "seeds": [42, 43, 44, 45, 46],
}

# Stream B: Diversity Specialists
# Trained on smaller crops to maximize patch diversity
STREAM_B_CONFIG = {
    "name": "StreamB_Diversity",
    "img_size": (160, 160),
    "seeds": [47, 48, 49, 50, 51],
}

# Augmentation Settings
AUGMENTATION_PARAMS = {
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.5,
    "rotate90_prob": 0.5,
}

# =============================================================================
# 4. MODEL ARCHITECTURE
# =============================================================================
# Architecture: Resolution-Preserved Deep U-Net
MODEL_CONFIG = {
    "encoder_filters": [32, 64, 128, 256, 512],
    "decoder_filters": [256, 128, 64, 32],
    # Structural Innovation:
    # Restrict downsampling to 8x (equivalent to 3-level) while keeping 4-level depth.
    # This is achieved by modifying the final encoder block.
    "bottleneck_block_index": 4,
    "bottleneck_stride": 1,  # Standard is 2
    "bottleneck_dilation": 2,  # Standard is 1
    "downsampling_factor": 8,  # Used for padding calculations
    "use_reflection_padding": True,
    "final_activation": "sigmoid",
}

# =============================================================================
# 5. TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 1000

# Optimization
OPTIMIZER_NAME = "Adam"
SCHEDULER_NAME = "CosineAnnealingLR"
SCHEDULER_T_MAX = 1000

# Debugging / Development
# Set to an integer (e.g., 100) to limit dataset size for fast debugging/prototyping.
# Set to None to use the full dataset.
DEBUG_MAX_SAMPLES = None

# =============================================================================
# 6. INFERENCE & SUBMISSION
# =============================================================================
# Test-Time Augmentation (TTA)
# D4 Group: 8 views (Original, Rot90, Rot180, Rot270 + Horizontal Flips of each)
TTA_ENABLED = True
TTA_VIEWS = 8

# Inference Padding
# Images must be padded to be multiples of this factor (matching model downsampling)
INFERENCE_PADDING_MULTIPLE = 8
