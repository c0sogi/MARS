import os
import torch
from pathlib import Path


class Config:
    # --- Reproducibility ---
    SEED = 42

    # --- Paths ---
    INPUT_DIR = Path("./input")
    METADATA_DIR = Path("./metadata")
    # Specific working directory for this idea
    WORKING_DIR = Path("./working/idea_10")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # --- Data Configuration ---
    # Input volume depth (z-axis)
    Z_DIM = 65

    # Patch size for training (Large context as per idea)
    PATCH_SIZE = 256

    # Stride for validation/inference sliding window
    # Using half patch size for overlap
    INFERENCE_STRIDE = PATCH_SIZE // 2

    # --- Model Architecture ---
    # Initial projection channels
    MODEL_CHANNELS = 64

    # Dilation rates for the sequential backbone
    DILATION_RATES = [1, 2, 4, 8, 16]

    # --- Training Hyperparameters ---
    # Batch size (Small batch size allowed for wider model)
    BATCH_SIZE = 8

    # Number of random patches to sample per epoch (Regularization)
    SAMPLES_PER_EPOCH = 12000

    # Optimizer settings
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Training duration
    NUM_EPOCHS = 30

    # Early stopping patience
    EARLY_STOPPING_PATIENCE = 5

    # --- Loss Configuration ---
    # Weight for the auxiliary boundary loss
    # L_total = L_mask + lambda * L_boundary
    AUX_LOSS_WEIGHT = 0.5

    # --- Inference & Post-processing ---
    # Threshold optimization range
    THRESHOLD_START = 0.2
    THRESHOLD_END = 0.8
    THRESHOLD_STEP = 0.05

    # Test Time Augmentation (TTA)
    # If True, averages predictions from flips and rotations
    USE_TTA = True

    # --- Hardware ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # Adjust based on vCPUs available (12 vCPUs total)

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print("Configuration:")
        print(f"  Model Channels: {cls.MODEL_CHANNELS}")
        print(f"  Patch Size: {cls.PATCH_SIZE}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Dilation Rates: {cls.DILATION_RATES}")
        print(f"  Samples Per Epoch: {cls.SAMPLES_PER_EPOCH}")
        print(f"  Aux Loss Weight: {cls.AUX_LOSS_WEIGHT}")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Working Dir: {cls.WORKING_DIR}")
        print("=" * 30)
