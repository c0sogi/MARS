import os
import torch
import numpy as np
import random


class Config:
    """
    Central configuration for the Stabilized Squeeze-and-Excitation Residual-Visual Network.
    Defines paths, hyperparameters, and structural constants.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for caching intermediate artifacts
    WORKING_DIR = "./working/idea_31"
    SUBMISSION_DIR = "./submission"

    # ==========================================
    # Data Processing / Feature Engineering
    # ==========================================
    # Temporal Window: t-5 to t+5 (11 frames total)
    WINDOW_SIZE = 11

    # Input Clamping Layer Bounds
    # Strictly clamps continuous features to prevent gradient destabilization
    CLAMP_MIN = -50.0
    CLAMP_MAX = 50.0

    # Video/Tracking Synchronization
    TRACKING_FREQ = 10  # Hz

    # ==========================================
    # Model Architecture
    # ==========================================
    # Entity Embeddings
    EMBEDDING_DIM = 32

    # Kinematic Backbone
    HIDDEN_DIM = 256
    SE_REDUCTION = 8  # Reduction ratio for Squeeze-and-Excitation blocks

    # Visual Stream
    VISUAL_HIDDEN_DIM = 64

    # Fusion & Regularization
    DROPOUT = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 2048
    LEARNING_RATE = 1e-3
    EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 3

    # Focal Loss Parameters (Critical for Class Imbalance)
    # Alpha=0.25 balances easy negatives; Gamma=2.0 focuses on hard examples
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    # ==========================================
    # Compute Resources
    # ==========================================
    NUM_WORKERS = 4  # Optimized for 12 vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Flags to control dataset size for rapid iteration
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50000

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary working and submission directories.
        2. Sets fixed random seeds for reproducibility across numpy, random, and torch.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def print_config(cls):
        """Prints the active configuration for verification."""
        print(f"\n{'='*10} Configuration {'='*10}")
        print(f"  Device:         {cls.DEVICE}")
        print(f"  Window Size:    {cls.WINDOW_SIZE} frames")
        print(f"  Clamping:       [{cls.CLAMP_MIN}, {cls.CLAMP_MAX}]")
        print(f"  Focal Loss:     Alpha={cls.FOCAL_ALPHA}, Gamma={cls.FOCAL_GAMMA}")
        print(f"  Batch Size:     {cls.BATCH_SIZE}")
        print(f"  Working Dir:    {cls.WORKING_DIR}")
        print(f"  Debug Mode:     {cls.DEBUG}")
        print(f"{'='*35}\n")
