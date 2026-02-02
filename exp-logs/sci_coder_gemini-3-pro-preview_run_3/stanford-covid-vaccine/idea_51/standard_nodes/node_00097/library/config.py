import os


class Config:
    """
    Central configuration for the RNA Degradation Prediction pipeline.
    Implements the 'Deep Stabilized Bias-Refined Decoupled BiGRU' strategy parameters.
    """

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68  # Number of scored positions (0 to 67)

    # Input Feature Channels:
    # 4 (Nucleotide One-Hot) + 3 (Structure One-Hot) + 7 (Loop Type One-Hot)
    INPUT_CHANNELS = 14

    # =========================================================================
    # Model Architecture Hyperparameters
    # =========================================================================
    # Backbone capacity maximized within safe limits
    HIDDEN_DIM = 384
    NUM_LAYERS = 4

    # Convolutional Stem
    CONV_FILTERS = 256
    KERNEL_SIZE = 3

    # Regularization
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    EPOCHS = 50
    PATIENCE = 10  # For Early Stopping

    # Gradient Clipping is mandatory for the 4-layer hybrid architecture
    GRAD_CLIP = 1.0

    SEED = 42
    NUM_WORKERS = 4

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_51"
    SUBMISSION_DIR = "./submission"

    # Metadata Inputs (Parquet format)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Processed Data Cache (Numpy format for speed/determinism)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_cache.npy")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_cache.npy")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_cache.npy")

    # Outputs
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Target Definitions
    # =========================================================================
    # All 5 targets present in the ground truth
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Subset of targets used for the MCRMSE metric calculation
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    @classmethod
    def setup_directories(cls):
        """Creates necessary working and submission directories if they don't exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup_directories()
