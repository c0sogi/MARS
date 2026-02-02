import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration module for the Translation-Invariant SegFormer (MiT-B2) Ink Detection Pipeline.
    """

    # =========================================================================
    # PATHS & DIRECTORIES
    # =========================================================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Paths (Pre-generated)
    METADATA_DIR = "./metadata"
    METADATA_TRAIN = os.path.join(METADATA_DIR, "train.csv")
    METADATA_VAL = os.path.join(METADATA_DIR, "validation.csv")
    METADATA_TEST = os.path.join(METADATA_DIR, "test.csv")

    # Output & Caching
    WORKING_DIR = "./working"
    # Specific cache directory for this experimental idea
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_19")
    # Submission output path (Home directory as per instructions)
    SUBMISSION_PATH = "./submission.csv"

    # Ensure cache directory exists immediately upon config load
    os.makedirs(CACHE_DIR, exist_ok=True)

    # =========================================================================
    # REPRODUCIBILITY
    # =========================================================================
    SEED = 42

    @staticmethod
    def set_seed(seed=42):
        """Sets fixed random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    # =========================================================================
    # COMPUTE RESOURCES
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # =========================================================================
    # DATA GENERATION & PREPROCESSING
    # =========================================================================
    TILE_SIZE = 512
    STRIDE = 512  # Non-overlapping stride for training data generation

    # Z-Slice Configuration: "Overlapping Thick Slab"
    # Input tensor shape: (3, 512, 512)
    # Each channel represents a sub-slab of thickness 12.
    # Sub-slabs overlap by 50% (6 slices).
    Z_SLICE_CHANNELS = 3
    SLAB_THICKNESS = 12
    SLAB_OVERLAP = 6

    # Safe-Zone Multi-View Sampling
    # We define 3 discrete views for training to enforce translation invariance.
    # Values represent the START Z-index for the 24-slice block.
    # View A (High):   Start 16 -> Covers 16-40 (Ch0:16-28, Ch1:22-34, Ch2:28-40)
    # View B (Center): Start 20 -> Covers 20-44 (Ch0:20-32, Ch1:26-38, Ch2:32-44)
    # View C (Low):    Start 24 -> Covers 24-48 (Ch0:24-36, Ch1:30-42, Ch2:36-48)
    TRAIN_VIEWS = {"A": 16, "B": 20, "C": 24}

    # =========================================================================
    # MODEL HYPERPARAMETERS
    # =========================================================================
    MODEL_ARCH = "mit_b2"  # SegFormer B2 Backbone
    DECODER_TYPE = "MLP"

    # Micro-Dataset Optimization Protocol
    BATCH_SIZE = 8  # Strictly 8 to prevent underfitting
    LEARNING_RATE = 6e-5  # Conservative LR for stability
    EPOCHS = 15  # Sufficient for convergence on small dataset

    # =========================================================================
    # EVALUATION & METRICS
    # =========================================================================
    BETA = 0.5  # For F0.5 Score (Precision weighted higher)
    SMOOTH = 1e-6  # Smoothing factor for Dice calculation
    THRESHOLD = 0.5  # Binary classification threshold

    # Validation Gating
    # Only generate submission if Validation F0.5 > BASELINE_SCORE
    BASELINE_SCORE = 0.598


# Apply seeding immediately upon import
Config.set_seed(Config.SEED)
