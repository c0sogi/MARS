import os
import json
import hashlib


class Config:
    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------------------------------
    WINDOW_SIZE = 15  # Sliding window size
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    PATIENCE = 10  # For Early Stopping

    # Model Architecture
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    NHEAD = 4
    DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # Data Processing & Features
    # -------------------------------------------------------------------------
    # Approximate conversion factor for degrees to meters
    # Note: Longitude scaling technically varies with latitude, but we use a
    # fixed scaler for feature normalization input. The reconstruction step
    # should use the precise local factor.
    DEG_TO_M_LAT = 111320.0

    # Raw columns to load from GNSS files
    RAW_GNSS_COLS = [
        "utcTimeMillis",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "Cn0DbHz",
        "RawPseudorangeUncertaintyMeters",
        "SvElevationDegrees",
        "SvAzimuthDegrees",
    ]

    # Derived Kinematic Features (Sequence Input)
    # These are calculated relative to the window center
    KINEMATIC_FEATURES = [
        "rel_lat_m",  # Relative Latitude in meters
        "rel_lon_m",  # Relative Longitude in meters
        "rel_alt_m",  # Relative Altitude in meters
        "delta_lat_m",  # Velocity/Delta Latitude
        "delta_lon_m",  # Velocity/Delta Longitude
        "delta_alt_m",  # Velocity/Delta Altitude
        "cn0_scaled",  # Standardized Signal Strength
        "unc_scaled",  # Standardized Uncertainty
    ]

    # Derived Sky/Environmental Features (Context Input)
    # Aggregated statistics over the window
    SKY_FEATURES = [
        "mean_cn0",
        "std_cn0",
        "mean_unc",
        "mean_elev",
        "std_elev",
        "mean_azim",  # Note: Circular mean handling required in preprocessing
        "sat_count",  # Number of satellites
    ]

    # Targets
    # We predict the residual (Ground Truth - Baseline WLS) in meters
    TARGET_COLS = ["dLat_m", "dLon_m"]

    @classmethod
    def get_config_hash(cls):
        """
        Generates a unique hash based on the configuration parameters that affect
        data processing. This is used for cache safety.
        """
        config_dict = {
            "window_size": cls.WINDOW_SIZE,
            "kinematic_features": cls.KINEMATIC_FEATURES,
            "sky_features": cls.SKY_FEATURES,
            "target_cols": cls.TARGET_COLS,
            "deg_to_m_lat": cls.DEG_TO_M_LAT,
            "raw_gnss_cols": cls.RAW_GNSS_COLS,
        }

        # Serialize to JSON string with sorting to ensure determinism
        config_str = json.dumps(config_dict, sort_keys=True)

        # Create MD5 hash
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()

    @classmethod
    def get_cache_path(cls, prefix):
        """
        Returns a file path for caching data, incorporating the config hash.
        e.g., ./working/idea_15/train_data_<hash>.parquet
        """
        config_hash = cls.get_config_hash()
        filename = f"{prefix}_{config_hash}.parquet"
        return os.path.join(cls.WORKING_DIR, filename)

    @classmethod
    def get_npy_cache_path(cls, prefix):
        """
        Returns a file path for caching numpy arrays.
        """
        config_hash = cls.get_config_hash()
        filename = f"{prefix}_{config_hash}.npy"
        return os.path.join(cls.WORKING_DIR, filename)
