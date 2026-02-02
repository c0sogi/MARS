import os
import torch

# =============================================================================
# PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_8"
SUBMISSION_PATH = "./submission.csv"

# Ensure the working directory exists for saving checkpoints and cache
os.makedirs(WORKING_DIR, exist_ok=True)

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
TILE_SIZE = 512

# --- Volumetric Z-Slice Strategy: Overlapping Thick Slabs ---
# We extract 3 channels of Maximum Intensity Projections (MIPs).
# Each slab has a depth of SLAB_SIZE.
# The slabs overlap by 50% (STRIDE = SLAB_SIZE // 2).
# Base Configuration:
#   Channel 1: Slices 20 to 32 (Start 20)
#   Channel 2: Slices 26 to 38 (Start 26)
#   Channel 3: Slices 32 to 44 (Start 32)
Z_START = 20
SLAB_SIZE = 12
STRIDE = 6
NUM_CHANNELS = 3

# --- Volumetric Augmentation ---
# Z-Jitter: During training, randomly shift the Z_START by +/- Z_JITTER_RANGE.
# e.g., if range is 2, start index can be between 18 and 22.
Z_JITTER_RANGE = 2

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# SegFormer with Mix Transformer B3 backbone
BACKBONE = "mit_b3"
PRETRAINED = True
IN_CHANNELS = 3
NUM_CLASSES = 1

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEED = 42
BATCH_SIZE = 16  # Fits comfortably on A100-40GB with MiT-B3
LEARNING_RATE = 2e-4  # Standard fine-tuning rate for Transformers
NUM_EPOCHS = 30  # Sufficient for convergence on small dataset (412 patches)

# Validation Gating: Only generate submission if F0.5 score exceeds this baseline.
VALIDATION_THRESHOLD = 0.588

# =============================================================================
# COMPUTE RESOURCES
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Utilizing available vCPUs (12 available)
NUM_WORKERS = 8
