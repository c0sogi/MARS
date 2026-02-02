import os
import torch


class Config:
    """
    Configuration for the Zoneout-Regularized Wide-Stream Residual BiGRU model.
    Centralizes all hyperparameters, file paths, and model specifications.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SUBSET_SIZE = 100  # Number of samples to use in debug mode

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Metadata (Parquet files generated in ./metadata)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Original Input (for submission format reference)
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output & Working Directories
    # Using 'idea_30' to isolate this experiment's artifacts
    WORKING_DIR = "./working/idea_30"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Artifact Paths
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    SCORED_LEN = 68

    # The 3 columns actually scored in the competition
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Input Vocabularies
    VOCAB_SIZE = 4  # Nucleotides: A, G, C, U
    LOOP_TYPES = 7  # Predicted Loop Types: S, M, I, B, H, E, X

    # =========================================================================
    # Model Architecture
    # (Zoneout-Regularized Wide-Stream Residual BiGRU)
    # =========================================================================
    EMBED_DIM = 128  # High-Dimensional Embeddings to prevent bottlenecks
    HIDDEN_DIM = 512  # Wide-Stream capacity (W=512)
    NUM_LAYERS = 6  # Shallow and Wide backbone
    ZONEOUT_PROB = 0.1  # Internal regularization for RNN dynamics

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 20
    BATCH_SIZE = 32  # Optimized for A100 GPU
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Low weight decay; relying on Zoneout for regularization
    MAX_GRAD_NORM = 1.0

    # Scheduler Settings (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # =========================================================================
    # Hardware & Execution
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup_directories(cls):
        """Ensures that the working and cache directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)


# Initialize directories on module import
Config.setup_directories()
