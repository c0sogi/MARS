import os
import torch
import random
import numpy as np


class Config:
    # --- General ---
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    # --- Data Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VALID_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching and checkpoints
    # Using idea_14 as implied by the context of the iteration
    WORKING_DIR = "./working/idea_14"
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission output
    SUBMISSION_PATH = "./submission.csv"

    # --- Data Processing ---
    TILE_SIZE = 512
    STRIDE = 512  # Matches metadata generation

    # Z-Axis Configuration for Siamese Views
    # We define 3 views: High, Center, Low
    # Each view uses the "Overlapping Thick Slab" projection with 3 channels

    # Starting Z-indices for the 3 views
    # View 1 (High): Start N=16
    # View 2 (Center): Start N=20
    # View 3 (Low): Start N=24
    VIEW_START_INDICES = [16, 20, 24]

    # Channel configuration relative to the View Start Index (N)
    # Channel 1: N to N+12
    # Channel 2: N+6 to N+18
    # Channel 3: N+12 to N+24
    # We store these as (start_offset, end_offset) tuples
    CHANNEL_OFFSETS = [
        (0, 12),  # Channel 1
        (6, 18),  # Channel 2
        (12, 24),  # Channel 3
    ]

    # Normalization
    PIXEL_MIN = 0.0
    PIXEL_MAX = 65535.0

    # --- Model Hyperparameters ---
    MODEL_BACKBONE = "mit_b2"
    PRETRAINED = True
    IN_CHANNELS = 3  # Standard RGB interface per Siamese branch
    NUM_CLASSES = 1  # Binary segmentation

    # --- Training Hyperparameters ---
    BATCH_SIZE = 16  # A100 can handle this for B2 512x512
    LEARNING_RATE = 6e-5
    WEIGHT_DECAY = 1e-2
    EPOCHS = 15

    # Scheduler
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MIN_DELTA = 0.001

    # Logic Gate for Submission
    # Only generate submission if Val F0.5 > BASELINE_SCORE
    BASELINE_SCORE = 0.598

    @classmethod
    def setup(cls):
        """
        Sets up the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        torch.cuda.manual_seed_all(cls.SEED)

        # Deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set TF32 for A100 to speed up training without significant precision loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    @classmethod
    def get_slice_ranges(cls, view_idx):
        """
        Helper to get the absolute slice ranges for a specific view index (0, 1, or 2).
        Returns a list of 3 tuples: [(start, end), (start, end), (start, end)]
        """
        base_z = cls.VIEW_START_INDICES[view_idx]
        ranges = []
        for offset_start, offset_end in cls.CHANNEL_OFFSETS:
            ranges.append((base_z + offset_start, base_z + offset_end))
        return ranges
