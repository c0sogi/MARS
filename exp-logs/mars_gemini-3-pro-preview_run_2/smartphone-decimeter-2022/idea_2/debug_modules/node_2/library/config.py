import os


class Config:
    # -------------------------------------------------------------------------
    # Directories and Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory for this specific idea/experiment to store processed parquets
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_2")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Submission Path
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing Configuration
    # -------------------------------------------------------------------------
    # Window size for the sequence model (number of time steps/epochs)
    # The model will see a sequence of length WINDOW_SIZE centered on the target time.
    WINDOW_SIZE = 30

    # Features derived from GNSS data to be used as input to the model.
    # These features are computed during the preprocessing stage (aggregation & differencing).
    INPUT_FEATURES = [
        "SatelliteCount",  # Number of satellites visible/used
        "MeanCn0",  # Average Carrier-to-Noise density
        "MeanUncertainty",  # Average Raw Pseudorange Uncertainty
        "DeltaLat",  # Rate of change in Latitude (from WLS baseline)
        "DeltaLon",  # Rate of change in Longitude (from WLS baseline)
        "DeltaAlt",  # Rate of change in Altitude (from WLS baseline)
    ]

    # Columns representing the baseline WLS position.
    # These are used to calculate the target residuals and to reconstruct the final path.
    BASELINE_COLS = ["WlsLat", "WlsLon", "WlsAlt"]

    # Target columns for the regression task.
    # These represent the distance in meters between the Ground Truth and the Baseline WLS.
    TARGET_COLUMNS = ["DeltaEast", "DeltaNorth"]

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    RANDOM_STATE = 42

    # Training parameters
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 8

    # Bi-LSTM Architecture parameters
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2
