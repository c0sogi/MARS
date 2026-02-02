import os
import torch


class Config:
    """
    Configuration for RNA Degradation Prediction using Deep Bias-Refined Decoupled BiGRU.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SUBSET_SIZE = 100  # Number of samples to use when DEBUG is True

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_42"

    # Input Data (Parquet Metadata)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Submission Example
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Cache File Names (to be used by dataset module)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_cache.npy")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_cache.npy")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_cache.npy")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    SEQ_SCORED = 68

    # Input Feature Dimensions
    # 4 (Sequence: A,G,C,U) + 3 (Structure: (,),.) + 7 (Loop: S,M,I,B,H,E,X)
    INPUT_CHANNELS = 14

    # Target Columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Columns used for the competition metric
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Convolutional Stem
    STEM_KERNEL_SIZE = 3
    STEM_FILTERS = 256

    # Deep Backbone (BiGRU + Structural Interaction)
    HIDDEN_DIM = 384
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 64
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Optimization Stability
    MAX_GRAD_NORM = 1.0  # Gradient clipping threshold

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 7
    MIN_DELTA = 1e-4

    # =========================================================================
    # Compute Resources
    # =========================================================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup_directories(cls):
        """Ensures the working directory exists."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)


# Initialize directories on module import
Config.setup_directories()
