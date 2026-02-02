import os
import torch

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# Reproducibility
SEED = 42

# Compute Environment
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 12 vCPUs available; setting workers to a safe number for data loading
NUM_WORKERS = 4

# =============================================================================
# FILE PATHS
# =============================================================================


class PATHS:
    """Central repository for all file paths."""

    INPUT = "./input"
    TRAIN_FRAGMENTS = os.path.join(INPUT, "train")
    TEST_FRAGMENTS = os.path.join(INPUT, "test")

    # Pre-generated metadata
    METADATA_DIR = "./metadata"
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching processed slabs and model checkpoints
    # Specific to this idea/experiment
    WORKING_DIR = "./working/idea_24"

    # Output directory for submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")


# Ensure necessary writable directories exist
os.makedirs(PATHS.WORKING_DIR, exist_ok=True)
os.makedirs(PATHS.SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# DATASET GENERATION PARAMETERS
# =============================================================================

# Input Dimensions
TILE_SIZE = 512
Z_DIM = 3  # Input channels (RGB-like interface)

# Overlapping Stratified Depth Projection
# Each specialist sees a Z-window of 24 slices (e.g., 16-40).
# We project this 24-slice volume into 3 channels using overlapping slabs.
# Logic:
#   Channel 0: slices [0:12]
#   Channel 1: slices [6:18]
#   Channel 2: slices [12:24]
# This corresponds to Thickness=12, Stride=6 (50% overlap).
SLAB_PARAMS = {"thickness": 12, "stride": 6}

# Specialist Definitions
# Defines the exact Z-range in the source volume for each model.
SPECIALIST_SETTINGS = {
    "High": {"z_start": 16, "z_end": 40, "description": "Targeting upper ink layers"},
    "Mid": {"z_start": 20, "z_end": 44, "description": "Targeting middle ink layers"},
    "Low": {"z_start": 24, "z_end": 48, "description": "Targeting lower ink layers"},
}

# Augmentation Policy
# Strictly geometric to preserve radiodensity semantics.
AUGMENTATION_PARAMS = {
    "geometric": True,  # HorizontalFlip, VerticalFlip, RandomRotate90
    "z_jitter": False,  # Explicitly excluded to maintain depth consistency
    "intensity": False,  # Explicitly excluded to preserve physical values
}

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

MODEL_PARAMS = {
    "encoder_name": "mit_b2",  # SegFormer B2 (~25M params)
    "encoder_weights": "imagenet",  # Pretrained on ImageNet
    "in_channels": Z_DIM,  # 3 channels
    "classes": 1,  # Binary segmentation
    "activation": None,  # Return logits for BCEWithLogitsLoss
    "decoder_type": "MLP",  # Standard SegFormer decoder
}

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

TRAINING_PARAMS = {
    "batch_size": 8,  # Small batch size for gradient frequency
    "learning_rate": 6e-5,  # Conservative LR for stability
    "epochs": 15,  # Sufficient for convergence on small data
    "optimizer": "AdamW",
    "weight_decay": 1e-2,
    "scheduler": "CosineAnnealingLR",
    "min_lr": 1e-6,
    "valid_threshold": 0.55,  # Minimum F0.5 score to save a checkpoint
    "loss": "BceDiceLoss",  # Balanced BCE + Dice
    "use_amp": True,  # Automatic Mixed Precision
    "clip_grad": 1.0,  # Gradient clipping
    "debug": False,  # If True, runs on a tiny subset
}

# =============================================================================
# INFERENCE PARAMETERS
# =============================================================================

INFERENCE_PARAMS = {
    "threshold": 0.5,  # Probability threshold for binary mask
    "batch_size": 8,  # Inference batch size
    "tta": False,  # Test Time Augmentation (optional)
}
