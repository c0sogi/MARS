import os
import torch


class Config:
    """
    Central configuration for the Feature-Based Deep Regression Network.
    Defines file paths, hyperparameters, and compute settings.
    """

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------
    SEED = 42

    # ---------------------------------------------------------
    # File Paths
    # ---------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "mlp_model.pth")
    SCALER_SAVE_PATH = os.path.join(WORKING_DIR, "scaler.npy")

    # Cache Paths for Processed Features
    TRAIN_FEATURES_CACHE = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_CACHE = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_CACHE = os.path.join(WORKING_DIR, "test_features.parquet")

    # ---------------------------------------------------------
    # Data Settings
    # ---------------------------------------------------------
    NUM_SENSORS = 10
    # Debugging: Set to a small integer (e.g., 100) to limit dataset size for fast checking
    DEBUG_SAMPLE_SIZE = None

    # ---------------------------------------------------------
    # Compute Settings
    # ---------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Using 4 workers as a safe default given 12 vCPUs
    NUM_WORKERS = 4

    # ---------------------------------------------------------
    # Model Hyperparameters
    # ---------------------------------------------------------
    # Architecture
    INPUT_DIM = 70  # 10 sensors * 7 stats (mean, std, min, max, skew, kurt, nans) - approximate, will be calculated dynamically
    HIDDEN_LAYERS = [256, 128, 64]
    DROPOUT_RATE = 0.3

    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 8

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
