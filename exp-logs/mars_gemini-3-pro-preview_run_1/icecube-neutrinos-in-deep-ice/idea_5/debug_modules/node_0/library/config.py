import os
import torch


class Config:
    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea (checkpoints, cache)
    WORKING_DIR = "./working/idea_5"

    # Metadata Paths
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.parquet")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.parquet")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.parquet")

    # Geometry Path
    GEOMETRY_PATH = os.path.join(INPUT_DIR, "sensor_geometry.csv")

    # Submission Path
    # The generic requirement is ./submission/submission.csv
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Data Processing & Sampling
    # ==========================================
    # Physics-Informed Sampling parameters
    MAX_PULSES = 196  # N: Total number of pulses to sample per event
    EARLY_PULSES = 48  # K: Number of earliest pulses (temporal priority) to keep
    # The remaining (N - K) pulses are sampled based on charge weights

    # Normalization Constants (derived from Data Analysis)
    # Time: mean~13k, std~4.4k, max~100k. Scale to roughly [-1, 1] or [0, 1] range.
    TIME_SCALE = 30000.0
    # Coordinates: Detector is roughly 1000m x 1000m x 1000m.
    COORD_SCALE = 500.0
    # Charge: Log1p transform is usually applied, so no scalar scale needed

    # Loader settings
    NUM_WORKERS = 12  # Using available vCPUs
    PIN_MEMORY = True

    # ==========================================
    # 3. Model Architecture (Spatiotemporal Point Transformer)
    # ==========================================
    MODEL_DIM = 128  # Embedding dimension
    NUM_HEADS = 4  # Number of attention heads
    NUM_LAYERS = 4  # Number of Transformer Encoder layers
    DIM_FEEDFORWARD = 256  # FFN hidden dimension
    DROPOUT = 0.1  # Dropout rate

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 256  # Fits comfortably on A100 with N=196

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 1

    # Training Loop Control
    EPOCHS = 10
    # The dataset has ~118M events. Training on all is too slow for the time limit.
    # We define a subset size to train on a representative sample per epoch.
    TRAIN_SUBSET_SIZE = 5_000_000  # Train on 5M samples per epoch
    VAL_SUBSET_SIZE = 200_000  # Validate on 200k samples

    PATIENCE = 3  # Early stopping patience

    # ==========================================
    # 5. Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 6. Setup Logic
    # ==========================================
    @classmethod
    def setup(cls):
        """Ensure necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducible seeds
        import numpy as np
        import random

        torch.manual_seed(cls.SEED)
        np.random.seed(cls.SEED)
        random.seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)


# Execute setup immediately when module is imported to ensure directories exist
Config.setup()
