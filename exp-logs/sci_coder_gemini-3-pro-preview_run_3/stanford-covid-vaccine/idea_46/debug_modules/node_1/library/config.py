import os
import torch


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Implements the 'Deep Stabilized Bias-Refined Decoupled BiGRU' strategy settings.
    """

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_46"  # Specific directory for this run/idea
    SUBMISSION_DIR = "./submission"

    # Input Metadata Files (Parquet format)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Sample Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Architecture: Deep Stabilized Bias-Refined Decoupled BiGRU
    HIDDEN_DIM = 384  # High capacity backbone
    NUM_LAYERS = 4  # Deep 4-layer stack
    KERNEL_SIZE = 3  # For the 1D Convolutional Stem
    DROPOUT = 0.1  # Regularization

    # Dimensions
    SEQ_LEN = 107  # Total sequence length
    SEQ_SCORED = 68  # Scored positions
    NUM_TARGETS = 5  # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    # Input Features
    # 4 (ACGU) + 3 (Structure: .()) + 7 (Loop: SMIBHEX) = 14
    INPUT_CHANNELS = 14

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32  # Adjusted for A100 40GB and 384 dim model
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 25  # Sufficient for convergence
    EARLY_STOPPING_PATIENCE = 7

    # Stability
    MAX_GRAD_NORM = 1.0  # Mandatory for deep hybrid architecture

    # =========================================================================
    # Data Processing & Caching
    # =========================================================================
    LOAD_CACHED_DATA = True  # Flag to control loading from cache
    NUM_WORKERS = 4  # Number of DataLoader workers

    # Cache file names
    TRAIN_CACHE_FILE = os.path.join(WORKING_DIR, "train_cache.npy")
    VAL_CACHE_FILE = os.path.join(WORKING_DIR, "val_cache.npy")
    TEST_CACHE_FILE = os.path.join(WORKING_DIR, "test_cache.npy")

    # =========================================================================
    # Debugging
    # =========================================================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SUBSET_SIZE = 100

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def create_directories(cls):
        """
        Creates necessary directories for working files and submissions.
        Should be called at the start of the pipeline.
        """
        # Update dependent paths to reflect any runtime changes to WORKING_DIR
        cls.TRAIN_CACHE_FILE = os.path.join(cls.WORKING_DIR, "train_cache.npy")
        cls.VAL_CACHE_FILE = os.path.join(cls.WORKING_DIR, "val_cache.npy")
        cls.TEST_CACHE_FILE = os.path.join(cls.WORKING_DIR, "test_cache.npy")
        cls.BEST_MODEL_PATH = os.path.join(cls.WORKING_DIR, "best_model.pth")

        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
