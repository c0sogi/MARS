import os


class Config:
    # -------------------------------------------------------------------------
    # Directories and Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Dedicated cache directory for this specific idea/experiment
    # Using 'idea_22' as the identifier for this run
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_22")

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Sample Submission Path
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    # Time alignment: 1000ms = 1Hz
    TIME_QUANTIZATION_MS = 1000

    # Stratification Rules
    # Stratum 1: Global (All satellites) - Implicit base
    # Stratum 2: High Precision (L5/E5a/B2a)
    L5_SIGNAL_TYPES = ["GPS_L5", "GAL_E5A", "BDS_B2A", "QZS_J5", "GLO_G3"]
    # Stratum 3: High Risk (Low Elevation)
    LOW_ELEVATION_THRESHOLD = 30.0  # Degrees

    # Aggregation Statistics for each stratum
    # Applied to: Cn0DbHz, SvElevationDegrees
    STRATA_STATS = ["mean", "std", "min", "max"]

    # Global Context Features (Scalar values per timestamp)
    GLOBAL_FEATURES = [
        "SignalCount",
        "RawPseudorangeUncertaintyMeters_mean",
        "AzimuthCentroid_X",  # Cosine component of signal-weighted azimuth
        "AzimuthCentroid_Y",  # Sine component of signal-weighted azimuth
    ]

    # -------------------------------------------------------------------------
    # Model Hyperparameters (1D ResUNet with ASPP)
    # -------------------------------------------------------------------------
    # Input Channels Calculation:
    # 3 Strata * 2 Variables (Cn0, Elev) * 4 Stats (Mean, Std, Min, Max) = 24
    # + 4 Global Features
    # = 28 Channels
    IN_CHANNELS = 28

    # Encoder/Decoder Settings
    NUM_FILTERS = 64  # Base number of filters
    KERNEL_SIZE = 3  # Convolution kernel size
    ENCODER_DEPTH = 4  # Number of downsampling blocks

    # Bottleneck Settings
    ASPP_DILATIONS = [1, 6, 12, 18]  # Atrous rates for multi-scale context

    # Regularization
    DROPOUT_RATE = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32  # Number of full drive sequences per batch
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Decimated Deep Supervision
    # Auxiliary heads attached to decoder layers.
    # Strides correspond to the downsampling factor at that decoder level.
    # Weights determine contribution to total loss.
    DEEP_SUPERVISION_STRIDES = [2, 4, 8]
    AUX_LOSS_WEIGHTS = [0.3, 0.2, 0.1]  # Weights for stride 2, 4, 8 respectively

    # Gradient Clipping
    GRAD_CLIP_NORM = 1.0

    # -------------------------------------------------------------------------
    # WGS84 Constants (For Coordinate Conversion)
    # -------------------------------------------------------------------------
    WGS84_A = 6378137.0  # Semi-major axis
    WGS84_B = 6356752.314245  # Semi-minor axis
    # Flattening factor f = (a-b)/a is approx 1/298.257223563

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    # If True, runs on a small subset of drives for rapid testing
    DEBUG = False
    DEBUG_DRIVE_COUNT = 5

    @classmethod
    def setup(cls):
        """
        Ensures all necessary writeable directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(
            f"Configured directories:\n  Cache: {cls.CACHE_DIR}\n  Submission: {cls.SUBMISSION_DIR}"
        )
