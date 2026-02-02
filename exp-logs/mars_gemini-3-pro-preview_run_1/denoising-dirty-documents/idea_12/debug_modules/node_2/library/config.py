import os
import torch


class Config:
    """
    Central configuration for the Signal-Aligned Reflection-Padded Ensemble.
    Defines hyperparameters, file paths, and architectural constants.
    """

    # --- Directory Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea (Idea 12)
    WORKING_DIR = "./working/idea_12"
    SUBMISSION_DIR = "./submission"

    # --- File Paths ---
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Preprocessing ---
    # Patch size for training crops (supports 4-level U-Net receptive field)
    PATCH_SIZE = 320
    # Signal Inversion: Maps background (white) to 0, text (black) to 1
    INVERT_SIGNAL = True
    # Padding mode: 'reflect' ensures boundary continuity for noise statistics
    PADDING_MODE = "reflect"

    # --- Model Architecture ---
    # Standard 4-Level U-Net
    MODEL_NAME = "UNet_4Level_SignalAligned"
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    START_FILTERS = 32
    DEPTH = 4

    # --- Training Hyperparameters ---
    # Massive Converged Bagging: 10 independent models
    ENSEMBLE_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    # Training duration and optimization
    NUM_EPOCHS = 1000
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    SCHEDULER_T_MAX = 1000  # For Cosine Annealing

    # --- Compute Resources ---
    # Utilizing available vCPUs (safe margin below 12)
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Inference ---
    # Test-Time Augmentation: D4 Group (8 views)
    TTA_VIEWS = 8

    @classmethod
    def initialize(cls):
        """
        Sets up the necessary directories for the experiment.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories initialized: {cls.WORKING_DIR}, {cls.SUBMISSION_DIR}")

    @staticmethod
    def get_model_path(seed):
        """Returns the file path for a saved model checkpoint based on its seed."""
        return os.path.join(Config.WORKING_DIR, f"model_seed_{seed}.pth")
