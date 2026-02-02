import os
import torch
from pathlib import Path


class Config:
    """
    Configuration for the Dual-Frame Causal Graph Network (DF-CGN).
    """

    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------
    INPUT_DIR = Path("./input")
    METADATA_DIR = Path("./metadata")
    WORKING_DIR = Path("./working/idea_7")

    # Specific file paths
    SENSOR_GEOMETRY_PATH = INPUT_DIR / "sensor_geometry.csv"
    TRAIN_METADATA_PATH = METADATA_DIR / "train_metadata.parquet"
    VAL_METADATA_PATH = METADATA_DIR / "val_metadata.parquet"
    TEST_METADATA_PATH = METADATA_DIR / "test_metadata.parquet"
    SAMPLE_SUBMISSION_PATH = INPUT_DIR / "sample_submission.csv"

    # Output paths
    MODEL_SAVE_PATH = WORKING_DIR / "model.pth"
    SUBMISSION_PATH = WORKING_DIR / "submission.csv"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # Data Processing
    # ---------------------------------------------------------
    # Number of pulses to sample per event
    MAX_PULSES = 196

    # Spatiotemporal distance scaling factor (alpha)
    # Used in k-NN: dist = sqrt(|dx|^2 + alpha * |dt|^2)
    # Speed of light is ~0.3 m/ns.
    # If we want 10 ns to be comparable to 3 meters, alpha ~ (3/10)^2 = 0.09
    # Setting a value to balance spatial and temporal locality.
    TIME_SCALE_ALPHA = 0.1

    # Feature Scaling / Normalization Constants (Approximate IceCube dimensions)
    # These are used to normalize raw coordinates to [-1, 1] range roughly
    POS_SCALE = 600.0  # Meters
    TIME_SCALE = 30000.0  # Nanoseconds (relative to event start)
    CHARGE_SCALE = 5.0  # Log10(charge) scaling

    # ---------------------------------------------------------
    # Model Architecture
    # ---------------------------------------------------------
    # Input Channels:
    # 1. Raw x (normalized)
    # 2. Raw y (normalized)
    # 3. Raw z (normalized)
    # 4. Relative Time (normalized)
    # 5. Canonical x' (normalized)
    # 6. Canonical y' (normalized)
    # 7. Canonical z' (normalized)
    # 8. Log10(Charge)
    # 9. Auxiliary (0 or 1)
    IN_CHANNELS = 9

    # Backbone settings
    HIDDEN_DIM = 128
    LATENT_DIM = 256
    K_NEIGHBORS = 20
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # ---------------------------------------------------------
    # Training Hyperparameters
    # ---------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 256  # A100 has 40GB, can handle large batches
    NUM_WORKERS = 12  # Matches available vCPUs

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 30

    # Scheduler (OneCycleLR)
    PCT_START = 0.3
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 1000.0

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5

    # ---------------------------------------------------------
    # System / Runtime
    # ---------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debug flag to run on a smaller subset of data for rapid testing
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50000
