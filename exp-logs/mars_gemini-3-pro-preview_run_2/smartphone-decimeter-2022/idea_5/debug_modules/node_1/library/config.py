import os


class Config:
    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for this specific idea (idea_5)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_5")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    # Sliding window size (number of epochs). Should be odd to have a center.
    # 11 epochs corresponds to roughly +/- 5 seconds context.
    WINDOW_SIZE = 11

    # Raw GNSS columns to load for aggregation
    RAW_GNSS_COLS = [
        "utcTimeMillis",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "Cn0DbHz",
        "RawPseudorangeUncertaintyMeters",
        "Svid",
    ]

    # Columns to aggregate per epoch
    # We take the first WLS position as the baseline for that epoch
    # We take mean of signal metrics and count of satellites
    AGG_COLS = {
        "WlsPositionXEcefMeters": "first",
        "WlsPositionYEcefMeters": "first",
        "WlsPositionZEcefMeters": "first",
        "Cn0DbHz": "mean",
        "RawPseudorangeUncertaintyMeters": "mean",
        "Svid": "count",
    }

    # Renaming map for aggregated columns
    AGG_RENAME = {
        "Cn0DbHz": "MeanCn0",
        "RawPseudorangeUncertaintyMeters": "MeanUncertainty",
        "Svid": "SatCount",
    }

    # Features to be used as input to the model (per timestep in the window)
    # These are computed during the windowing process
    INPUT_FEATURES = [
        # Local Shape (Relative Position to Window Center)
        "rel_east",
        "rel_north",
        "rel_up",
        # Dynamics (First order differences / Velocities)
        "delta_east",
        "delta_north",
        "delta_up",
        # Signal Quality
        "MeanCn0",
        "MeanUncertainty",
        "SatCount",
    ]

    # Number of input channels for the 1D CNN
    NUM_FEATURES = len(INPUT_FEATURES)

    # Target columns (Residuals in Meters: GT - Baseline)
    TARGET_COLS = ["target_east", "target_north"]

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # Convolutional Backbone
    CNN_CHANNELS = [64, 128, 256]  # Channels for each conv layer
    KERNEL_SIZE = 3

    # Prediction Head (MLP)
    HIDDEN_DIM = 128
    DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # Training Parameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 30
    PATIENCE = 5  # Early stopping patience

    # Debugging / Development
    DEBUG = False  # Set to True to use a smaller subset of data
    SAMPLE_SIZE = 1000  # Number of trips to sample if DEBUG is True

    # Random Seed
    SEED = 42
