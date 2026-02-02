import os
import torch


class Config:
    # =========================================================================
    # 1. File Paths and Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_23")
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # 2. Data Processing & Feature Engineering
    # =========================================================================
    # Random Seed
    SEED = 42

    # Signal Stratification Criteria
    # High Precision Signals (L5/E5a/B2a bands)
    L5_SIGNAL_TYPES = ["GPS_L5", "GAL_E5A", "BDS_B2A", "QZS_J5"]
    # High Risk Signals (Low Elevation)
    HIGH_RISK_ELEV_THRESHOLD = 30.0  # degrees

    # Strata Definitions
    STRATA = ["Global", "HighPrecision", "HighRisk"]

    # Aggregation Statistics for Stratified Features
    # Applied to Cn0DbHz and SvElevationDegrees
    STATS_FUNCS = ["mean", "std", "min", "max"]

    # Carrier Phase Validity Bit (Bit 0 of AccumulatedDeltaRangeState)
    ADR_STATE_VALID_BIT = 1  # 1 << 0

    # Feature Dimensions Calculation
    # Per Stratum:
    #   - Cn0DbHz (4 stats)
    #   - SvElevationDegrees (4 stats)
    #   - PhaseValid (Count, Fraction) (2 features)
    #   Total per stratum = 10
    FEATS_PER_STRATUM = 10

    # Global Context Features:
    #   - SignalCount
    #   - RawPseudorangeUncertaintyMeters (mean)
    #   - AzimuthCentroid_X
    #   - AzimuthCentroid_Y
    GLOBAL_FEATS_COUNT = 4

    # Total Input Channels
    INPUT_CHANNELS = (len(STRATA) * FEATS_PER_STRATUM) + GLOBAL_FEATS_COUNT

    # Target Transformation
    # We predict residuals in meters (East, North) relative to WLS baseline
    OUTPUT_CHANNELS = 2

    # =========================================================================
    # 3. Model Architecture (1D SE-ResUNet)
    # =========================================================================
    # Encoder channel progression
    ENCODER_CHANNELS = [32, 64, 128, 256]

    # Decoder channel progression (reverse of encoder usually, but can be defined explicitly)
    DECODER_CHANNELS = [256, 128, 64, 32]

    # Squeeze-and-Excitation Reduction Ratio
    SE_REDUCTION = 16

    # ASPP Dilations for the bottleneck
    ASPP_DILATIONS = [1, 6, 12, 18]

    # Deep Supervision
    # Auxiliary heads attached to decoders.
    # Weights for loss summation: [Deepest Decoder, ..., Final Output]
    # We have 4 decoder stages (corresponding to encoder depths).
    # Final output is at full resolution.
    AUX_LOSS_WEIGHTS = [0.1, 0.2, 0.3, 1.0]

    # =========================================================================
    # 4. Training Hyperparameters
    # =========================================================================
    EPOCHS = 30
    BATCH_SIZE = 16  # Number of drives/sequences per batch
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP_VAL = 1.0

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 8

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # =========================================================================
    # 5. Debugging / Development
    # =========================================================================
    # Set to True to run on a small subset of drives for testing the pipeline
    DEBUG = False
    DEBUG_DRIVE_COUNT = 2
