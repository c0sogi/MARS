import os
import torch


class Config:
    # --- Reproducibility ---
    SEED = 42

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_21"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Model Checkpoint
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission File
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Processing ---
    # Signal Groups for Multi-Frequency Aggregation
    # Group A: Standard Precision (L1 Band)
    SIGNAL_TYPES_L1 = ["GPS_L1", "GAL_E1", "GLO_G1", "BDS_B1I", "BDS_B1C", "QZS_J1"]

    # Group B: High Precision (L5 Band)
    SIGNAL_TYPES_L5 = ["GPS_L5", "GAL_E5A", "BDS_B2A", "QZS_J5"]

    # Features to calculate stats for (Mean, Std, Min, Max)
    # Applied to each signal group separately
    STAT_FEATURES = ["Cn0DbHz", "SvElevationDegrees"]

    # Global features (not split by signal group)
    GLOBAL_FEATURES = ["RawPseudorangeUncertaintyMeters", "Az_X", "Az_Y"]

    # WLS Baseline columns
    BASELINE_COLS = [
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
    ]

    # --- Model Architecture ---
    IN_CHANNELS = (
        (len(SIGNAL_TYPES_L1) > 0) * len(STAT_FEATURES) * 4
        + (len(SIGNAL_TYPES_L5) > 0) * len(STAT_FEATURES) * 4
        + len(GLOBAL_FEATURES)
        + 1
    )  # +1 for Satellite Count (Global)
    # Note: The exact calculation depends on the preprocessing implementation details,
    # but this provides a configuration basis.

    # ResUNet Hyperparameters
    ENCODER_CHANNELS = [64, 128, 256, 512]
    DECODER_CHANNELS = [256, 128, 64, 32]
    ASPP_DILATIONS = [1, 6, 12, 18]

    # Auxiliary Loss Weights
    AUX_LOSS_WEIGHT = 0.4

    # --- Training ---
    DEBUG = False  # Set to True to use a small subset of data
    EPOCHS = 50
    BATCH_SIZE = 8  # Smaller batch size for full sequences
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # Early stopping patience
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Gradient Clipping
    MAX_GRAD_NORM = 1.0

    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories
Config.setup()
