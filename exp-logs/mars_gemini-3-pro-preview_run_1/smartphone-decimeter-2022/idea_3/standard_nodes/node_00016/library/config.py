import os
import torch


class Config:
    """
    Configuration class for the Deep-Set TCN smartphone location prediction model.
    Stores file paths, feature definitions, model hyperparameters, and training settings.
    """

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"

    # Ensure the working directory exists for caching
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output submission path
    SUBMISSION_PATH = "./submission/submission.csv"

    # =========================================================================
    # Feature Definitions
    # =========================================================================
    # Raw columns to load from the device_gnss.csv files
    GNSS_COLS = [
        "utcTimeMillis",
        "Cn0DbHz",
        "SvElevationDegrees",
        "SvAzimuthDegrees",
        "RawPseudorangeUncertaintyMeters",
        "ConstellationType",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]

    # Satellite-specific features to be processed by the Deep Set Encoder
    # Note: Azimuth will be transformed to sin/cos, ConstellationType to embedding/one-hot
    SAT_FEATURES = [
        "Cn0DbHz",
        "SvElevationDegrees",
        "SvAzimuthDegrees",
        "RawPseudorangeUncertaintyMeters",
        "ConstellationType",
    ]

    # Global features context to be concatenated after Deep Set aggregation
    # SatCount is derived from the number of rows per epoch
    # WlsAlt (Altitude) can be derived from WLS ECEF coordinates
    GLOBAL_FEATURES = ["SatCount", "WlsAlt"]

    # Target columns (Ground Truth)
    TARGET_COLS = ["LatitudeDegrees", "LongitudeDegrees"]

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Deep Set Encoder
    MAX_SATELLITES = (
        32  # Maximum number of satellites per epoch (fixed size for tensor)
    )
    SAT_HIDDEN_DIM = 64  # Hidden dimension for satellite MLP
    SAT_EMBEDDING_DIM = 64  # Dimension of the aggregated epoch embedding

    # TCN Backbone
    # Input dim = Satellite Embedding + Number of Global Features
    # The data loader produces 8 aggregated features manually.
    TCN_INPUT_DIM = 8
    TCN_CHANNELS = 64  # Number of channels in TCN layers
    TCN_KERNEL_SIZE = 3  # Kernel size for dilated convolutions
    TCN_LAYERS = 4  # Number of residual blocks in TCN
    TCN_DROPOUT = 0.1  # Dropout rate

    # Output Head
    OUTPUT_DIM = 2  # Predicts residuals: Delta Latitude, Delta Longitude

    # =========================================================================
    # Training & Optimization
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32  # Batch size (number of sequences)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 20  # Maximum number of training epochs
    PATIENCE = 5  # Early stopping patience
    NUM_WORKERS = 4  # Number of DataLoader workers
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Data Processing & Caching
    # =========================================================================
    DEBUG = False  # Set to True to use a small subset of data for debugging
    DEBUG_SIZE = 100  # Number of samples to use in debug mode
    CACHE_DATA = True  # Whether to save/load processed tensors from disk
