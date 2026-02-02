import os
import torch
import numpy as np
import random

# =============================================================================
# FILE SYSTEM PATHS
# =============================================================================
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TRAIN_CLEANED_DIR = os.path.join(INPUT_DIR, "train_cleaned")
TEST_DIR = os.path.join(INPUT_DIR, "test")

METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working directory for checkpoints and cache
WORKING_DIR = "./working/idea_10"
os.makedirs(WORKING_DIR, exist_ok=True)

CACHE_DIR = os.path.join(WORKING_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Submission directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# SYSTEM & HARDWARE
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Optimized for 12 vCPUs
SEED = 42


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic mode ensures reproducibility but may impact performance slightly
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# DATA PIPELINE CONFIGURATION
# =============================================================================
# Signal Alignment: Invert intensity so text=1 (signal) and background=0 (sparse).
# This aligns zero-padding with the background.
INVERT_INTENSITY = True

# Patch Sizes for the Heterogeneous Dual-Stream Strategy
STREAM_A_PATCH_SIZE = 320  # Large context for structure
STREAM_B_PATCH_SIZE = 160  # Small context for texture diversity

# =============================================================================
# MODEL ARCHITECTURE CONSTANTS
# =============================================================================
# Stream A: Context Specialist (4-Level U-Net)
STREAM_A_CONFIG = {
    "name": "Context_Specialist",
    "input_channels": 1,
    "output_channels": 1,
    "depth": 4,
    "encoder_filters": [32, 64, 128, 256, 512],
    "decoder_filters": [256, 128, 64, 32],
    "patch_size": STREAM_A_PATCH_SIZE,
}

# Stream B: Texture Specialist (3-Level U-Net)
STREAM_B_CONFIG = {
    "name": "Texture_Specialist",
    "input_channels": 1,
    "output_channels": 1,
    "depth": 3,
    "encoder_filters": [32, 64, 128, 256],
    "decoder_filters": [128, 64, 32],
    "patch_size": STREAM_B_PATCH_SIZE,
}

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
EPOCHS = 1000
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
T_MAX = 1000  # For Cosine Annealing Scheduler

# Ensemble Configuration: Number of independent models to train per stream type.
# Total models = 2 * ENSEMBLE_SIZE
ENSEMBLE_SIZE = 10

# =============================================================================
# DEBUGGING & UTILITIES
# =============================================================================
DEBUG = False
DEBUG_SUBSET_SIZE = 50  # Number of samples to use if DEBUG is True


def get_config(debug=False, overrides=None):
    """
    Retrieves the configuration dictionary, optionally enabling debug mode
    or applying specific overrides.

    Args:
        debug (bool): If True, enables debug settings (e.g., subset size).
        overrides (dict, optional): Dictionary of parameters to override.

    Returns:
        dict: A dictionary containing the active configuration.
    """
    cfg = {
        "input_dir": INPUT_DIR,
        "working_dir": WORKING_DIR,
        "device": DEVICE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "t_max": T_MAX,
        "ensemble_size": ENSEMBLE_SIZE,
        "stream_a": STREAM_A_CONFIG,
        "stream_b": STREAM_B_CONFIG,
        "invert_intensity": INVERT_INTENSITY,
        "debug": debug,
        "subset_size": DEBUG_SUBSET_SIZE if debug else None,
    }

    if overrides:
        cfg.update(overrides)

    return cfg
