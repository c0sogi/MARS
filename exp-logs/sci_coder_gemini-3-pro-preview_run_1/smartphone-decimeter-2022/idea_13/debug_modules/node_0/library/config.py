import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_13"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # -------------------------------------------------------------------------
    # Data Parameters
    # -------------------------------------------------------------------------
    SEED = 42

    # WGS84 Constants for coordinate conversion (Degrees <-> Meters)
    WGS84_A = 6378137.0
    WGS84_F = 1 / 298.257223563

    # Feature Engineering Definitions
    # Global features derived from aggregation of all visible satellites
    GLOBAL_FEATURES = [
        "Cn0DbHz_mean",
        "Cn0DbHz_std",
        "Cn0DbHz_min",
        "Cn0DbHz_max",
        "SvElevationDegrees_mean",
        "SvElevationDegrees_std",
        "SvElevationDegrees_min",
        "SvElevationDegrees_max",
        "SatCount",
        "RawPseudorangeUncertaintyMeters_mean",
    ]

    # Directional features (4 quadrants * 2 metrics)
    # Quadrants: NE (0-90), SE (90-180), SW (180-270), NW (270-360)
    # These capture the geometry of signal blockage
    DIRECTIONAL_FEATURES = [
        "NE_SatCount",
        "NE_Cn0DbHz_mean",
        "SE_SatCount",
        "SE_Cn0DbHz_mean",
        "SW_SatCount",
        "SW_Cn0DbHz_mean",
        "NW_SatCount",
        "NW_Cn0DbHz_mean",
    ]

    FEATURE_NAMES = GLOBAL_FEATURES + DIRECTIONAL_FEATURES
    NUM_FEATURES = len(FEATURE_NAMES)

    # Targets: Delta East, Delta North (Meters relative to WLS baseline)
    NUM_CLASSES = 2

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # 1D ResUNet Configuration
    MODEL_NAME = "ResUNet1D_ASPP"
    NUM_FILTERS = 32  # Base number of filters in the first encoder block
    ASPP_DILATIONS = [1, 6, 12, 24]  # Dilation rates for the bottleneck

    # Deep Supervision Weights for Loss Calculation
    # Loss = 1.0 * Final_Output_Loss + 0.5 * Aux_Output_Loss
    DEEP_SUPERVISION_WEIGHTS = {"final": 1.0, "aux": 0.5}

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    TRAIN_WINDOW_SIZE = 256  # Window size (timesteps) for training crops
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging
    DEBUG = False  # Set to True to use a small subset of data for rapid testing
    DEBUG_SAMPLE_SIZE = 100
