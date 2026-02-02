import os
import torch


class Config:
    # --- Reproducibility ---
    SEED = 42

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    OUTPUT_DIR = "./submission"

    # Specific cache directory for this idea
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Model Save Path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_SAVE_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

    # --- Data Parameters ---
    # Input is (6 positions, 273 frequency bins, 256 time steps)
    # We vertically stack the 6 positions into a single channel image: (1, 1638, 256)
    # This transforms the problem into detecting a broken line pattern across the image.
    INPUT_SHAPE = (1, 1638, 256)
    NUM_CLASSES = 1

    # --- Compute Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Using 4 workers as a safe default for data loading given 12 vCPUs
    NUM_WORKERS = 4
    PIN_MEMORY = True if torch.cuda.is_available() else False

    # --- Training Hyperparameters ---
    BATCH_SIZE = 32
    EPOCHS = 8
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler settings (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.1
    SCHEDULER_PATIENCE = 2

    # Early Stopping settings
    EARLY_STOPPING_PATIENCE = 3

    # --- Debugging ---
    # If True, limits the dataset size for rapid testing
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 1000

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for working, caching, and submission.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)

        # Set deterministic behavior for CuDNN if using CUDA
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
