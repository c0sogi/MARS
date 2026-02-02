import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the Volcano Eruption Prediction task.
    Implements the 'Channel-Adaptive Hybrid EfficientNet' strategy parameters.
    """

    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    PROJECT_NAME = "volcano_eruption_prediction"
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Metadata
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Data Artifacts (Cache)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Scaler Artifacts
    TARGET_MEAN_PATH = os.path.join(WORKING_DIR, "target_mean.npy")
    TARGET_STD_PATH = os.path.join(WORKING_DIR, "target_std.npy")
    STATS_SCALER_MEAN_PATH = os.path.join(WORKING_DIR, "stats_scaler_mean.npy")
    STATS_SCALER_SCALE_PATH = os.path.join(WORKING_DIR, "stats_scaler_scale.npy")

    # Model Artifacts
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    # Sensor Data: 60001 samples / 10 mins (600s) ≈ 100 Hz
    SAMPLING_RATE = 100

    # Spectrogram Generation
    # N_MELS=128 matches EfficientNet spatial hierarchy
    # N_FFT=1024 / HOP=256 provides good time-frequency resolution
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 256
    TOP_DB = 80  # Dynamic range for AmplitudeToDB

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE = "resnet18"
    PRETRAINED = True
    IN_CHANNELS = 10  # 10 Seismic Sensors
    NUM_CLASSES = 1  # Regression output

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    EPOCHS = 35
    PATIENCE = 20  # Early stopping patience

    # Debugging
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200

    @classmethod
    def setup(cls):
        """
        Initialize the experiment environment:
        1. Create necessary directories.
        2. Set random seeds for reproducibility.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        cls.seed_everything(cls.SEED)

    @staticmethod
    def seed_everything(seed: int):
        """
        Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
