import os
import random
import numpy as np
import torch


class Config:
    """
    Central configuration for the Seismic Eruption Prediction task.
    Handles file paths, data parameters, model hyperparameters, and reproducibility settings.
    """

    # ---------------------------------------------------------
    # Directories
    # ---------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # File Paths
    # ---------------------------------------------------------
    # Metadata
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")

    # Scalers and Statistics (Saved as .npy for fast loading)
    # Target Scaling
    TARGET_MEAN_PATH = os.path.join(WORKING_DIR, "target_mean.npy")
    TARGET_STD_PATH = os.path.join(WORKING_DIR, "target_std.npy")

    # Feature Scaling (for the MLP branch)
    STATS_SCALER_MEAN_PATH = os.path.join(WORKING_DIR, "stats_scaler_mean.npy")
    STATS_SCALER_SCALE_PATH = os.path.join(WORKING_DIR, "stats_scaler_scale.npy")

    # Spectrogram Normalization (Global stats for the CNN branch)
    SPEC_MEAN_PATH = os.path.join(WORKING_DIR, "spec_mean.npy")
    SPEC_STD_PATH = os.path.join(WORKING_DIR, "spec_std.npy")

    # Feature Cache Paths (Parquet)
    # These store the engineered statistical features to avoid re-computation
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # ---------------------------------------------------------
    # Data Parameters
    # ---------------------------------------------------------
    NUM_SENSORS = 10
    SIGNAL_LENGTH = 60001
    SAMPLING_RATE = 100  # Hz (60001 samples over 10 minutes)

    # ---------------------------------------------------------
    # Spectrogram Parameters (Branch 1)
    # ---------------------------------------------------------
    N_FFT = 1024
    HOP_LENGTH = 256  # Resulting time dimension: ~235
    N_MELS = 64  # Resulting frequency dimension: 64
    F_MIN = 0
    F_MAX = None  # Defaults to Nyquist frequency

    # ---------------------------------------------------------
    # Model Architecture
    # ---------------------------------------------------------
    BACKBONE = "resnet18"
    PRETRAINED = True
    # MLP Hidden Layers (Branch 2)
    MLP_HIDDEN_DIMS = [128, 64]
    DROPOUT = 0.5

    # ---------------------------------------------------------
    # Training Hyperparameters
    # ---------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 50
    PATIENCE = 15  # Early stopping patience
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4  # For DataLoader

    @staticmethod
    def set_seed(seed=42):
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
