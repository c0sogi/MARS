import os
import torch


class Config:
    """
    Configuration class for the Deep Internally-Normalized Channel-Gated BiGRU (DIN-CG-BiGRU) model.
    Centralizes all file paths, model hyperparameters, and training settings.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # File Paths
    # ==========================================
    # Input Metadata (Parquet files)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Original Input (for submission format reference)
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output Directories
    WORKING_DIR = "./working/idea_30"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Output Files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Data Specifications
    # ==========================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Feature Dimensions
    # Sequence (A, G, U, C) -> 4
    # Structure (., (, )) -> 3
    # Predicted Loop Type (S, M, I, B, H, E, X) -> 7
    # Total Input Channels = 14
    INPUT_DIM = 14

    # Target Columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    NUM_TARGETS = 5

    # ==========================================
    # Model Architecture (DIN-CG-BiGRU)
    # ==========================================
    HIDDEN_DIM = 384  # As specified: High capacity
    NUM_LAYERS = 4  # As specified: Deep backbone
    CONV_FILTERS = 256  # Convolutional stem filters
    CONV_KERNEL_SIZE = 3  # Convolutional stem kernel
    DROPOUT = 0.1  # Regularization

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 16  # Conservative batch size for 384 dim * 4 layers
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    PATIENCE = 10  # Early stopping patience

    # Stability
    MAX_GRAD_NORM = 1.0  # Mandatory gradient clipping

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # Data loading workers

    # ==========================================
    # Debugging / Development
    # ==========================================
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100  # Number of samples to use in debug mode

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)


# Execute setup on module import to ensure directories exist
Config.setup()
