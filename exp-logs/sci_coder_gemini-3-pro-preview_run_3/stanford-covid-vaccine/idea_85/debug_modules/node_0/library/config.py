import os


class Config:
    """
    Configuration module for the RNA Degradation Prediction task.
    Implements the 'High-Capacity Enhanced-Context Synthesis' strategy settings.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this strategy's cache
    WORKING_DIR = "./working/idea_85"
    SUBMISSION_DIR = "./submission"

    # Data File Paths (using pre-generated metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68  # Only the first 68 positions are scored

    # Input Features:
    # 4 (One-hot Nucleotide: A, G, U, C)
    # + 3 (One-hot Structure: (, ), .)
    # + 7 (One-hot Loop Type: S, M, I, B, H, E, X)
    INPUT_DIM = 14

    # Target Columns (All 5 used for Multi-Task Learning)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Scored Columns (Subset used for Metric Calculation)
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # =========================================================================
    # Model Hyperparameters (High-Capacity Enhanced-Context Synthesis)
    # =========================================================================
    # Backbone Configuration
    HIDDEN_DIM = 384  # Hidden dimension per direction (Total 768)
    NUM_LAYERS = 4  # Deep 4-layer backbone
    DROPOUT = 0.1  # Conservative regularization (0.1)

    # Convolutional Stem Configuration
    CONV_FILTERS = 256
    CONV_KERNEL = 3

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-3
    WEIGHT_DECAY = 1e-4

    # Stability & Optimization
    CLIP_GRAD = 1.0  # Mandatory gradient clipping for stability
    PATIENCE = 10  # Early stopping patience

    # Hardware & Reproducibility
    NUM_WORKERS = 4
    SEED = 42

    def __init__(self, **kwargs):
        """
        Initialize configuration.

        Args:
            **kwargs: Arbitrary keyword arguments to override default class attributes.
        """
        # Override defaults with provided kwargs
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # Ensure necessary directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)
