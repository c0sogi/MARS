import os
import torch


class Config:
    """
    Global configuration for the Canonical-Frame Dynamic Graph Network (CF-DGN) solution.
    Includes paths, data processing parameters, model hyperparameters, and training settings.
    """

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # File System Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_DIR = os.path.join(WORKING_DIR, "model")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Input Files
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.parquet")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.parquet")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.parquet")
    SENSOR_GEOMETRY = os.path.join(INPUT_DIR, "sensor_geometry.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    MODEL_PATH = os.path.join(MODEL_DIR, "cf_dgn_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    # Number of pulses to sample per event (Hybrid sampling: high charge + early time)
    N_PULSES = 128

    # Number of nearest neighbors for dynamic graph construction
    K_NEIGHBORS = 8

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # Input features: x, y, z (rotated), time (relative), charge, auxiliary
    # We will likely use x, y, z, t, q, aux (6 features) or a subset.
    # Based on the idea, we use coordinates and time. Let's allocate for standard features.
    IN_CHANNELS = 5  # x, y, z, time, charge (in canonical frame)

    HIDDEN_DIM = 128
    EMBED_DIM = 128

    # Output: 3D vector (x, y, z) representing direction
    OUTPUT_DIM = 3

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 256  # A100 40GB allows for larger batch sizes
    EPOCHS = 15  # Maximum number of epochs
    LEARNING_RATE = 1e-3  # Initial learning rate
    WEIGHT_DECAY = 1e-4  # Regularization

    # Scheduler settings (Cosine Annealing)
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 3  # Stop if validation loss doesn't improve for 3 epochs

    # -------------------------------------------------------------------------
    # Hardware & Runtime
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # -------------------------------------------------------------------------
    # Debug / Development Flags
    # -------------------------------------------------------------------------
    # If True, runs on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SIZE = 20000  # Number of events to use in debug mode

    @classmethod
    def setup_directories(cls):
        """
        Ensures that all necessary working directories exist.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
