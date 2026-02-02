import os
import torch

# Ensure the working directory exists as per requirements
os.makedirs("./working/idea_11", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


class Config:
    # -------------------------------------------------------------------------
    # 1. Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"

    # Metadata paths (already generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache file paths for processed data
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_data.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_data.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_data.parquet")

    # Scaler and Model paths
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler.json")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission output
    SUBMISSION_PATH = os.path.join("./submission", "submission.csv")

    # -------------------------------------------------------------------------
    # 2. Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    RANDOM_STATE = 42

    # Sliding window size (number of epochs).
    # 11 epochs corresponds to roughly +/- 5 seconds context.
    WINDOW_SIZE = 11

    # IMU Aggregation window in milliseconds (centered on GNSS epoch)
    # 1000ms means [t-0.5s, t+0.5s]
    IMU_AGG_WINDOW_MS = 1000

    # Feature Definitions

    # 1. GNSS Aggregated Features (per epoch)
    # Derived from raw GNSS measurements
    GNSS_FEATURES = [
        "MeanCn0",  # Average signal strength
        "MeanUncertainty",  # Average received time uncertainty
        "SatelliteCount",  # Number of satellites
        "MeanElevation",  # Average elevation of satellites
        "AzimuthSpread",  # Standard deviation of azimuths (proxy for sky visibility)
    ]

    # 2. IMU Aggregated Features (per epoch)
    # Derived from high-frequency IMU stream
    IMU_FEATURES = [
        "AccelMag_Mean",  # Mean of acceleration magnitude
        "AccelMag_Std",  # Std of acceleration magnitude
        "GyroMag_Mean",  # Mean of gyroscope magnitude
        "GyroMag_Std",  # Std of gyroscope magnitude
    ]

    # 3. Coordinate & Dynamic Features (per epoch in window)
    # Calculated relative to the window center
    COORD_FEATURES = [
        "RelLatMeters",  # Latitude difference from center (converted to meters)
        "RelLonMeters",  # Longitude difference from center (converted to meters)
        "RelAltMeters",  # Altitude difference from center
        "VelLatMeters",  # Velocity in Latitude direction (m/s)
        "VelLonMeters",  # Velocity in Longitude direction (m/s)
        "VelAltMeters",  # Velocity in Altitude direction (m/s)
    ]

    # Full feature list used by the model
    # Input shape will be (Batch, Window_Size, Input_Dim)
    FEATURE_NAMES = GNSS_FEATURES + IMU_FEATURES + COORD_FEATURES
    INPUT_DIM = len(FEATURE_NAMES)

    # Target columns (residuals in meters)
    TARGET_COLUMNS = ["DeltaEastMeters", "DeltaNorthMeters"]
    OUTPUT_DIM = 2

    # -------------------------------------------------------------------------
    # 3. Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 6
    WEIGHT_DECAY = 1e-4

    # -------------------------------------------------------------------------
    # 4. Model Architecture
    # -------------------------------------------------------------------------
    # Residual 1D CNN Backbone
    CNN_CHANNELS = [64, 128, 256]
    CNN_KERNEL_SIZE = 3
    CNN_DROPOUT = 0.2

    # Prediction Head (MLP)
    MLP_HIDDEN_DIMS = [512, 256]
    MLP_DROPOUT = 0.2

    # -------------------------------------------------------------------------
    # 5. Hardware & Misc
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging: Set to a small number to limit dataset size during development
    # Set to None for full training
    DEBUG_SAMPLE_SIZE = None
