import os
import torch


class Config:
    """
    Configuration class for the Ventilator Pressure Prediction task.
    Centralizes file paths, model hyperparameters, and training settings.
    """

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Specific cache directory for this idea
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    # Raw Data Files
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_CHECKPOINT = os.path.join(CACHE_DIR, "best_model.pth")
    SCALER_PATH = os.path.join(CACHE_DIR, "scaler.npy")
    PROCESSED_TRAIN_DATA = os.path.join(CACHE_DIR, "train_data.npy")
    PROCESSED_VAL_DATA = os.path.join(CACHE_DIR, "val_data.npy")
    PROCESSED_TEST_DATA = os.path.join(CACHE_DIR, "test_data.npy")

    # ==========================================
    # Data Configuration
    # ==========================================
    SEED = 42
    SEQ_LEN = 80  # Fixed breath length

    # Input Features:
    # [time_step, u_in, u_out, R, C, u_in_cumsum]
    INPUT_DIM = 6

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 1000  # Number of breaths to use when DEBUG is True

    # ==========================================
    # Model Architecture (Stacked Bi-LSTM)
    # ==========================================
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 512  # A100 has 40GB VRAM, can handle large batches
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
