import os
import torch


class Config:
    """
    Configuration for High-Resolution 1D ResNet with Phase-Aware Stratified Aggregation.
    """

    # =========================================================================
    # 1. Directory & File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_29"
    SUBMISSION_DIR = "./submission"

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # 2. Data Preprocessing & Feature Engineering
    # =========================================================================
    # Temporal Alignment
    TIME_QUANTIZATION_HZ = 1  # Round timestamps to nearest second

    # Phase-Aware Stratification Criteria
    # Stratum 1: Global (All visible satellites) - handled by base aggregation

    # Stratum 2: High-Integrity
    # Signals in this set are considered high quality (L5 band)
    HIGH_INTEGRITY_SIGNALS = {"GPS_L5", "GAL_E5A", "BDS_B2A", "QZS_J5"}
    # Accumulated Delta Range State bit for Valid Carrier Phase (Bit 0)
    # 1 << 0 = 1. Used to identify phase-locked signals.
    ADR_STATE_VALID_BIT = 1

    # Stratum 3: Low-Quality
    # Satellites likely to introduce noise/multipath
    LOW_QUALITY_ELEV_TH = 30.0  # Degrees
    LOW_QUALITY_CN0_TH = 25.0  # dB-Hz

    # Feature Aggregation
    # Statistics to compute for each stratum (Global, HighInt, LowQual)
    STRATUM_STATS_COLS = ["Cn0DbHz", "SvElevationDegrees"]
    STRATUM_STATS_OPS = ["mean", "std", "min", "max"]

    # Global context features (computed once per timestamp across all sats)
    GLOBAL_CONTEXT_COLS = [
        "SatCount",
        "RawPseudorangeUncertaintyMeters",  # Mean uncertainty
        "Azimuth_Sin",  # Signal-weighted centroid sine
        "Azimuth_Cos",  # Signal-weighted centroid cosine
    ]

    # Input Dimension Calculation:
    # 3 Strata * 2 Columns * 4 Stats = 24 features
    # + 4 Global Context features = 28 total input channels
    IN_CHANNELS = 28

    # =========================================================================
    # 3. Model Architecture (HR-1D-ResNet)
    # =========================================================================
    STEM_CHANNELS = 64

    # Parallel Streams Configuration
    # Resolution factors relative to input T (1 = T, 4 = T/4, 16 = T/16)
    RESOLUTION_FACTORS = [1, 4, 16]

    # Channel width for each parallel stream
    STREAM_CHANNELS = [64, 128, 256]

    # Network Depth
    NUM_STAGES = 4  # Number of fusion stages
    BLOCKS_PER_STAGE = 4  # Residual blocks per stream between fusions

    # Convolution parameters
    KERNEL_SIZE = 3
    DROPOUT = 0.1

    # Output Head
    OUT_DIM = 2  # Delta North, Delta East (Meters)

    # =========================================================================
    # 4. Training Hyperparameters
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Data Loading
    NUM_WORKERS = 4
    # Batch size represents number of full drive sequences (or large chunks) per batch
    BATCH_SIZE = 4

    # Optimization
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-2
    GRAD_CLIP_NORM = 1.0

    # Scheduler (Cosine Annealing)
    EPOCHS = 50
    ETA_MIN = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # Deep Supervision Loss Weights
    # Weights for [High-Res, Med-Res, Low-Res] output heads
    LOSS_WEIGHTS = [1.0, 0.5, 0.25]

    # =========================================================================
    # 5. Debugging & Caching
    # =========================================================================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50  # Number of drives to sample when DEBUG is True

    # Cache file paths for processed tensors
    CACHE_TRAIN = os.path.join(WORKING_DIR, "train_processed.parquet")
    CACHE_VAL = os.path.join(WORKING_DIR, "val_processed.parquet")
    CACHE_TEST = os.path.join(WORKING_DIR, "test_processed.parquet")

    @classmethod
    def setup(cls):
        """Ensures necessary working directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_feature_names(cls):
        """Returns the ordered list of feature names for the input tensor."""
        names = []
        # Order must match the preprocessing logic: Global -> HighInt -> LowQual
        strata = ["Global", "HighInt", "LowQual"]
        for s in strata:
            for col in cls.STRATUM_STATS_COLS:
                for op in cls.STRATUM_STATS_OPS:
                    names.append(f"{s}_{col}_{op}")
        names.extend(cls.GLOBAL_CONTEXT_COLS)
        return names
