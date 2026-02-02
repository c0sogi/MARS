import os
import torch
import numpy as np
import random


class CFG:
    # =============================================================================
    # General Setup
    # =============================================================================
    PROJECT_NAME = "KinematicGeometricResUNet"
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =============================================================================
    # Paths
    # =============================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_20"
    SUBMISSION_DIR = "./submission"

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Files (Parquet format for processed tensors)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =============================================================================
    # Data Preprocessing & Feature Engineering
    # =============================================================================
    # Input Features Configuration
    # These keys correspond to aggregated statistics of raw GNSS columns per second
    FEATURE_COLS = [
        # Signal Strength Statistics (Robustness against noise)
        "Cn0DbHz_mean",
        "Cn0DbHz_std",
        "Cn0DbHz_min",
        "Cn0DbHz_max",
        # Satellite Elevation Statistics (Multipath detection)
        "SvElevationDegrees_mean",
        "SvElevationDegrees_std",
        "SvElevationDegrees_min",
        "SvElevationDegrees_max",
        # Geometric Orientation (Constellation shape)
        "SvAzimuthDegrees_sin_mean",
        "SvAzimuthDegrees_cos_mean",
        # Kinematics (Doppler shift for motion vs noise distinction)
        "PseudorangeRateMetersPerSecond_mean",
        "PseudorangeRateMetersPerSecond_std",
        # Reliability / Metadata
        "RawPseudorangeUncertaintyMeters_mean",
        "SatCount",
    ]

    INPUT_DIM = len(FEATURE_COLS)  # 14 features
    OUTPUT_DIM = 2  # Delta East, Delta North (Meters) relative to WLS baseline

    # Sequence Length for 1D CNN (Time window in seconds)
    # 128 seconds provides sufficient context for trajectory smoothing
    SEQUENCE_LENGTH = 128

    # =============================================================================
    # Model Architecture (1D ResUNet with ASPP)
    # =============================================================================
    ENCODER_CHANNELS = [32, 64, 128, 256]
    DECODER_CHANNELS = [256, 128, 64, 32]

    # Atrous Spatial Pyramid Pooling rates to capture multi-scale context
    ASPP_DILATIONS = [1, 6, 12, 24]
    KERNEL_SIZE = 3
    DROPOUT_RATE = 0.1

    # Scaled Deep Supervision Weights
    # Weights for losses at different decoder scales [Output, Aux1, Aux2]
    # Aux1 is 1/2 resolution, Aux2 is 1/4 resolution
    LOSS_WEIGHTS = [1.0, 0.5, 0.25]

    # =============================================================================
    # Training Hyperparameters
    # =============================================================================
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP = 1.0
    PATIENCE = 8  # Early stopping patience

    # Scheduler (CosineAnnealingLR)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # =============================================================================
    # Debugging
    # =============================================================================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500  # Number of drives to sample in debug mode


def set_seed(seed=42):
    """Sets the seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# Ensure necessary directories exist
os.makedirs(CFG.WORKING_DIR, exist_ok=True)
os.makedirs(CFG.SUBMISSION_DIR, exist_ok=True)
