import os
import torch


class Config:
    """
    Configuration for the Cascaded 1D ResUNet with Scaled Deep Supervision pipeline.
    """

    # --- Directory Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea to support caching requirements
    WORKING_DIR = "./working/idea_17"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_OUTPUT_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache Files (Parquet format preferred over pickle)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

    # --- Reproducibility ---
    SEED = 42

    # --- Data Processing ---
    # Raw columns to load from device_gnss.csv for feature engineering
    RAW_GNSS_COLS = [
        "utcTimeMillis",
        "Cn0DbHz",
        "SvElevationDegrees",
        "SvAzimuthDegrees",
        "RawPseudorangeUncertaintyMeters",
        "Svid",
        "ConstellationType",
    ]

    # Final Feature Columns (Input to Model)
    # These capture signal distribution, geometry, and quality
    FEATURE_COLS = [
        "Cn0DbHz_mean",
        "Cn0DbHz_std",
        "Cn0DbHz_min",
        "Cn0DbHz_max",
        "SvElevationDegrees_mean",
        "SvElevationDegrees_std",
        "SvElevationDegrees_min",
        "SvElevationDegrees_max",
        "SvAzimuth_sin_mean",
        "SvAzimuth_cos_mean",
        "SignalWeighted_Azimuth_sin",
        "SignalWeighted_Azimuth_cos",
        "SignalWeighted_Elevation",
        "SatCount",
        "RawPseudorangeUncertaintyMeters_mean",
    ]

    # Target Columns (Regression)
    # We predict offsets in meters (East, North) relative to the WLS baseline
    # These are converted back to Lat/Lon during submission generation
    TARGET_COLS = ["dEast", "dNorth"]

    # Dimensions
    INPUT_DIM = len(FEATURE_COLS)
    OUTPUT_DIM = 2

    # Sequence Parameters
    # Data is quantized to 1Hz.
    # For training, we crop fixed-length sequences.
    TRAIN_WINDOW_SIZE = 128
    TRAIN_WINDOW_STRIDE = 64

    # --- Model Architecture (Cascaded 1D ResUNet) ---
    # Stage 1: Deep ResUNet with ASPP (Coarse Correction)
    STAGE1_ENCODER_FILTERS = [32, 64, 128, 256]
    STAGE1_DECODER_FILTERS = [256, 128, 64, 32]
    ASPP_RATES = [1, 6, 12, 18]  # Dilations for ASPP bottleneck

    # Stage 2: Shallow Refinement U-Net (Fine Correction)
    # Input is concatenation of original features + Stage 1 output
    STAGE2_ENCODER_FILTERS = [32, 64]
    STAGE2_DECODER_FILTERS = [64, 32]

    KERNEL_SIZE = 3
    DROPOUT = 0.1

    # --- Training Hyperparameters ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 30
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Optimization & Scheduling
    EARLY_STOPPING_PATIENCE = 8
    GRAD_CLIP_MAX_NORM = 1.0  # Gradient clipping to stabilize training
    T_MAX = EPOCHS  # For CosineAnnealingLR scheduler

    # Loss Weights
    # Total Loss = Loss_Stage2 + AUX_LOSS_WEIGHT * Sum(Aux_Losses_Stage1)
    AUX_LOSS_WEIGHT = 0.4

    # --- Debugging ---
    # Set True to run on a small subset of data for quick pipeline verification
    DEBUG = False

    @classmethod
    def create_directories(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically create directories when config is imported
Config.create_directories()
