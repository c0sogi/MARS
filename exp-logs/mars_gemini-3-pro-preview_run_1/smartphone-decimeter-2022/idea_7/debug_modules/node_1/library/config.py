import os
import torch


class Config:
    # =========================================================================
    # 1. File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data (parquet/npy)
    # Using 'idea_7' as specified in requirements
    WORKING_DIR = "./working/idea_7"
    os.makedirs(WORKING_DIR, exist_ok=True)

    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Sample submission path
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # 2. Data Processing & Feature Engineering
    # =========================================================================
    SEED = 42

    # Features to extract from raw GNSS logs
    RAW_FEATURES = ["Cn0DbHz", "SvElevationDegrees", "RawPseudorangeUncertaintyMeters"]

    # Aggregation statistics for the raw features
    # These create the boundary-aware representation (e.g., Cn0DbHz_min, Cn0DbHz_max, etc.)
    AGG_STATS = ["min", "max", "mean"]

    # Metadata features derived during aggregation
    META_FEATURES = ["SatCount"]

    # Input dimension calculation:
    # (Number of Raw Features * Number of Stats) + Number of Meta Features
    INPUT_DIM = (len(RAW_FEATURES) * len(AGG_STATS)) + len(META_FEATURES)

    # Target columns (Residuals in meters: East, North)
    TARGET_COLS = ["DeltaEast", "DeltaNorth"]
    OUTPUT_DIM = 2

    # =========================================================================
    # 3. Model Architecture (1D TransUNet)
    # =========================================================================
    # Encoder parameters
    ENCODER_CHANNELS = [10, 32, 64, 128]  # Input -> Layer1 -> Layer2 -> Layer3

    # Transformer Bottleneck parameters
    TRANSFORMER_EMBED_DIM = 128
    TRANSFORMER_NUM_HEADS = 4
    TRANSFORMER_NUM_LAYERS = 2
    TRANSFORMER_DIM_FEEDFORWARD = 512
    TRANSFORMER_DROPOUT = 0.1

    # Decoder parameters (symmetric to encoder)
    DECODER_CHANNELS = [128, 64, 32, 16]

    # =========================================================================
    # 4. Training Hyperparameters
    # =========================================================================
    # Sequence length for training windows
    # We use a fixed window size for batched training.
    # Test inference can handle variable lengths or sliding windows.
    WINDOW_SIZE = 256

    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Huber Loss parameter
    HUBER_DELTA = 1.0

    # Early Stopping
    PATIENCE = 8

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # =========================================================================
    # 5. Debugging
    # =========================================================================
    # Set to True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of drives to sample in debug mode
