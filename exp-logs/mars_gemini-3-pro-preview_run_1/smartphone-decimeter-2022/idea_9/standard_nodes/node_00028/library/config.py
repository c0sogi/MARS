import os
import torch


class Config:
    """
    Central configuration for the Hybrid 1D ResUNet-GRU GNSS localization model.
    Defines paths, feature engineering parameters, model architecture, and training settings.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Feature Engineering Hyperparameters
    # =========================================================================
    # Distributional Embeddings (Histograms)
    # We bin signal strength and elevation to capture the distribution of satellite quality
    CN0_BINS = 10
    CN0_RANGE = (10.0, 50.0)  # Signal strength range in dB-Hz

    ELEVATION_BINS = 10
    ELEVATION_RANGE = (0.0, 90.0)  # Satellite elevation in degrees

    # Constants for WGS84 coordinate conversion (Approximate for MTV area)
    # Used to convert regression targets from Degrees to Meters
    LAT_TO_M = 110946.2576
    LON_TO_M = 88900.0  # Approx at ~37.4 degrees latitude

    # =========================================================================
    # Model Architecture Hyperparameters
    # =========================================================================
    # 1D ResUNet structure
    # Input channels will be calculated dynamically based on feature engineering
    ENCODER_CHANNELS = [32, 64, 128, 256]
    DECODER_CHANNELS = [256, 128, 64, 32]
    KERNEL_SIZE = 3

    # RNN Bottleneck (Bi-directional GRU)
    GRU_HIDDEN_DIM = 256
    GRU_LAYERS = 2
    BIDIRECTIONAL = True

    # Regularization
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 8  # Processing full drives as sequences
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    NUM_WORKERS = 4

    # Loss function configuration
    LOSS_FN = "L1"  # Mean Absolute Error (Robust to outliers)

    # =========================================================================
    # Runtime Options
    # =========================================================================
    DEBUG = False  # Set to True to run on a small subset of drives for testing
    DEBUG_DRIVE_COUNT = 2  # Number of drives to use in debug mode

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
