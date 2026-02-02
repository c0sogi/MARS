import os
import torch


class Config:
    """
    Configuration for the Expanded-Capacity Projected Wide-Stream BiLSTM experiment.
    Acts as a central source of truth for hyperparameters, paths, and settings.
    """

    # =========================================================================
    # Paths and Directories
    # =========================================================================
    METADATA_DIR = "./metadata"
    INPUT_DIR = "./input"
    # Cache directory for deterministic data processing artifacts
    CACHE_DIR = "./working/idea_66"
    # Directory for saving submission files
    SUBMISSION_DIR = "./submission"
    # Path to the final submission file
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    # Full length of the RNA sequence
    SEQ_LEN = 107
    # Number of positions scored (first 68 bases)
    PRED_LEN = 68
    # The specific ground truth columns used for training and scoring
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # =========================================================================
    # Model Architecture Hyperparameters
    # =========================================================================
    # Input Embedding Dimensions
    INPUT_DIM_SEQ = 128  # Dimension for nucleotide identity (A, G, C, U)
    INPUT_DIM_LOOP = 64  # Dimension for predicted loop type
    INPUT_DIM_STRUCT = 64  # Dimension for signed sinusoidal pairing distance

    # Backbone Dimensions (Expanded-Capacity Projected Wide-Stream BiLSTM)
    STREAM_WIDTH = 384  # The width of the stable residual stream
    LSTM_HIDDEN_DIM = 384  # Hidden dimension per direction (Internal width = 768)
    NUM_LAYERS = 6  # Number of Inverted Bottleneck Blocks
    DROPOUT = 0.2  # Dropout probability applied after projection

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3  # Initial learning rate for AdamW
    WEIGHT_DECAY = 1e-4  # Low weight decay to preserve recurrent signals
    CLIP_GRAD = 1.0  # Gradient clipping norm for stability
    EPOCHS = 20  # Fixed number of training epochs

    # =========================================================================
    # System and Reproducibility
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 2  # Number of dataloader workers
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Debugging and Flexibility
    # =========================================================================
    DEBUG = False  # Flag to enable debug mode with smaller dataset
    DEBUG_SAMPLES = 100  # Number of samples to use when DEBUG is True

    @classmethod
    def initialize(cls):
        """
        Creates necessary directories for caching and submissions if they do not exist.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
