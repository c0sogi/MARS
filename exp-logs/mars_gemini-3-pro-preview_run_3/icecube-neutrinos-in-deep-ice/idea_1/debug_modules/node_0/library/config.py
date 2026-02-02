import os
import torch


class Config:
    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Raw Data
    SENSOR_GEOMETRY_PATH = os.path.join(INPUT_DIR, "sensor_geometry.csv")

    # Metadata
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.parquet")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.parquet")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.parquet")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ---------------------------------------------------------
    # Data Parameters
    # ---------------------------------------------------------
    # Input features: x, y, z, time (relative), charge (log), auxiliary
    INPUT_DIM = 6
    # Output: vector x, vector y, vector z
    OUTPUT_DIM = 3

    # Pulse Sampling
    NUM_PULSES = 128  # Fixed number of pulses per event (N)

    # Normalization Constants (approximate based on EDA)
    TIME_SCALE = 30000.0  # To scale time to roughly 0-1 or similar range
    COORD_SCALE = 600.0  # Detector is roughly 1km^3, coordinates in meters

    # ---------------------------------------------------------
    # Model Architecture
    # ---------------------------------------------------------
    HIDDEN_DIM = 256
    DROPOUT = 0.1

    # ---------------------------------------------------------
    # Training Hyperparameters
    # ---------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 1024
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 3  # For Early Stopping

    # Debugging / Development
    DEBUG = False
    DEBUG_SUBSET_SIZE = 10000  # Number of events to use if DEBUG is True

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------
    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
