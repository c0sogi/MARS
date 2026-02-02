import os
import torch


class Config:
    """
    Configuration class for Ventilator Pressure Prediction.
    Holds all hyperparameters, file paths, and global settings.
    """

    # ==========================
    # Reproducibility
    # ==========================
    SEED = 42

    # ==========================
    # File Paths
    # ==========================
    # Input Data (Metadata)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output Directories & Files
    WORKING_DIR = "./working/idea_1"
    CACHE_DIR = WORKING_DIR  # Directory to store cached numpy/parquet files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model.pth")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================
    # Data Parameters
    # ==========================
    # Each breath consists of 80 time steps
    SEQUENCE_LENGTH = 80

    # Input features:
    # Dynamic: time_step, u_in, u_out
    # Static (repeated): R, C
    FEATURE_COLS = ["time_step", "u_in", "u_out", "R", "C"]
    TARGET_COL = "pressure"

    # ==========================
    # Model Hyperparameters
    # ==========================
    INPUT_DIM = len(FEATURE_COLS)  # 5
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.1

    # ==========================
    # Training Hyperparameters
    # ==========================
    BATCH_SIZE = 512
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Early Stopping
    PATIENCE = 10

    # Scheduler (OneCycleLR or ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3

    # ==========================
    # System & Debugging
    # ==========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debug mode: if True, use a small subset of data for rapid testing
    DEBUG = False
    DEBUG_BREATHS = 1000  # Number of unique breaths to use in debug mode

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories if they don't exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
