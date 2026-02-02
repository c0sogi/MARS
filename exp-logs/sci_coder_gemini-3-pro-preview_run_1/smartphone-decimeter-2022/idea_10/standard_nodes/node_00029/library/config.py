import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Sample Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Data Processing Parameters
    # =========================================================================
    # Temporal quantization (rounding timestamps to nearest second)
    TIME_ROUNDING = "1s"

    # Raw GNSS columns to load
    GNSS_RAW_COLS = [
        "Cn0DbHz",
        "SvElevationDegrees",
        "SvAzimuthDegrees",
        "RawPseudorangeMeters",
        "RawPseudorangeUncertaintyMeters",
        "utcTimeMillis",
    ]

    # Aggregation strategies for 1Hz quantization
    # Map raw column -> list of statistics
    # "count" on Cn0DbHz serves as the Satellite Count feature
    AGGREGATION_MAP = {
        "Cn0DbHz": ["mean", "min", "max", "count"],
        "SvElevationDegrees": ["mean", "min", "max"],
        "SvAzimuthDegrees": [
            "mean"
        ],  # Simple mean for azimuth (could be improved with circular mean, but sticking to standard)
        "RawPseudorangeMeters": ["std"],  # Variability proxy
        "RawPseudorangeUncertaintyMeters": ["mean"],
    }

    # List of final feature names after aggregation (order matters for model input)
    # This must match the order produced by the data processor
    INPUT_FEATURES = [
        "Cn0DbHz_mean",
        "Cn0DbHz_min",
        "Cn0DbHz_max",
        "Cn0DbHz_count",
        "SvElevationDegrees_mean",
        "SvElevationDegrees_min",
        "SvElevationDegrees_max",
        "SvAzimuthDegrees_mean",
        "RawPseudorangeMeters_std",
        "RawPseudorangeUncertaintyMeters_mean",
    ]

    INPUT_DIM = len(INPUT_FEATURES)

    # Target columns (Residuals in Meters: East, North)
    TARGET_COLS = ["dLat_meters", "dLon_meters"]
    OUTPUT_DIM = 2

    # =========================================================================
    # Model Architecture (1D Hybrid TransUNet)
    # =========================================================================
    # ResNet Encoder Settings
    # Channels for each block in the encoder
    ENCODER_CHANNELS = [32, 64, 128, 256]

    # Transformer Bottleneck Settings
    TRANSFORMER_D_MODEL = 256  # Must match last encoder channel count
    TRANSFORMER_NHEAD = 8
    TRANSFORMER_NUM_LAYERS = 4
    TRANSFORMER_DIM_FEEDFORWARD = 1024
    TRANSFORMER_DROPOUT = 0.1

    # Decoder Settings
    # Channels for each block in the decoder (reverse order of encoder)
    DECODER_CHANNELS = [128, 64, 32]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    NUM_EPOCHS = 50
    BATCH_SIZE = 4  # Small batch size as sequences (drives) can be long
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 1e-4

    # Gradient Clipping
    GRAD_CLIP_NORM = 1.0

    # =========================================================================
    # Inference
    # =========================================================================
    # WGS84 Constants for coordinate conversion
    WGS84_A = 6378137.0
    WGS84_F = 1 / 298.257223563
    WGS84_B = WGS84_A * (1 - WGS84_F)
