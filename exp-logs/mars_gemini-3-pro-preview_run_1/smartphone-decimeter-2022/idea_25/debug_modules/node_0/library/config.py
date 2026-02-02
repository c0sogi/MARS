import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # 1. Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_25"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. General Setup
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of workers for DataLoader

    # -------------------------------------------------------------------------
    # 3. Data Processing & Feature Engineering
    # -------------------------------------------------------------------------
    # Physical Constants
    LIGHT_SPEED = 299792458.0

    # Stratification Definitions
    # Stratum 1: Global (All visible satellites)
    # Stratum 2: High-Precision (SignalType L5/E5a/B2a OR Valid Carrier Phase)
    # Stratum 3: High-Risk (Elevation < 30 degrees)
    STRATA = ["global", "high_precision", "high_risk"]

    # Features to aggregate per stratum
    # We calculate statistics for these raw GNSS fields
    STRATUM_RAW_FIELDS = ["Cn0DbHz", "SvElevationDegrees"]

    # Statistics to compute for each raw field in each stratum
    STRATUM_STATS = ["mean", "std", "min", "max"]

    # Global Context Features (Concatenated once per timestamp)
    # - SignalCount: Total number of visible satellites
    # - RawPseudorangeUncertaintyMeters_mean: Average uncertainty across all sats
    # - AzimuthCentroid: Signal-weighted azimuth centroid (geometric context)
    GLOBAL_FEATURES = [
        "SignalCount",
        "RawPseudorangeUncertaintyMeters_mean",
        "AzimuthCentroid",
    ]

    # Calculated Input Dimension
    # (Num Strata * Num Raw Fields * Num Stats) + Num Global Features
    # (3 * 2 * 4) + 3 = 24 + 3 = 27
    IN_CHANNELS = (len(STRATA) * len(STRATUM_RAW_FIELDS) * len(STRATUM_STATS)) + len(
        GLOBAL_FEATURES
    )

    # Output Dimension: Delta East (m), Delta North (m)
    OUT_CHANNELS = 2

    # Target Scaling (Optional, can be used to normalize regression targets)
    TARGET_SCALE = 1.0

    # -------------------------------------------------------------------------
    # 4. Model Architecture (Stratified 1D Attention ResUNet)
    # -------------------------------------------------------------------------
    MODEL_NAME = "Stratified1DAttentionResUNet"

    # Encoder depth and width
    ENCODER_FILTERS = [32, 64, 128, 256]  # 4 levels

    # Bottleneck
    USE_ASPP = True
    ASPP_DILATIONS = [1, 6, 12, 18]

    # Decoder
    USE_ATTENTION_GATES = True  # Key component for noise filtering

    # Supervision
    DEEP_SUPERVISION = True  # Enable auxiliary heads at decoder levels

    # -------------------------------------------------------------------------
    # 5. Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    GRADIENT_CLIPPING = 1.0

    # Sequence Handling
    # We train on fixed-length random crops of the drives
    TRAIN_SEQUENCE_LENGTH = 256

    # Inference Handling
    # Overlap for sliding window inference (if needed) or full drive processing
    INFERENCE_OVERLAP = 64

    # Optimization
    EARLY_STOPPING_PATIENCE = 10

    # Scheduler (Cosine Annealing)
    COSINE_T_MAX = EPOCHS
    COSINE_ETA_MIN = 1e-6
