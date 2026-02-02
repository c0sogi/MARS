import os
import torch


class Config:
    """
    Configuration for the Periodic Granular Transformer with [CLS] Readout.
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True for fast debugging with subset of data

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"

    # Input files (using metadata splits as requested)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output directories and files
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    ID_COL = "id"
    TARGET_COL = "target"
    SEQUENCE_COL = "f_27"

    # Columns to exclude from numerical features (f_27 is handled separately)
    IGNORE_COLS = ["id", "target", "source_path", "f_27"]

    # Sequence Processing (f_27)
    # f_27 is a 10-character string. We use character-level tokenization.
    SEQ_LEN = 10
    VOCAB_SIZE = 40  # Sufficient for A-Z + Special Tokens (PAD, UNK, CLS if needed)

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # Transformer Encoder
    EMBED_DIM = 256  # Model dimension (must be divisible by NUM_HEADS)
    NUM_HEADS = 8
    NUM_LAYERS = 4
    FORWARD_EXPANSION = 4  # FFN hidden dim factor (256 * 4 = 1024)
    DROPOUT = 0.1

    # Readout Head ([CLS] -> MLP)
    HEAD_HIDDEN_DIM = 512
    NUM_CLASSES = 1  # Binary classification

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # Batching
    # A100 allows for large batch sizes
    BATCH_SIZE = 2048

    # Optimization
    EPOCHS = 30
    LEARNING_RATE = 1e-3  # Max LR for OneCycle
    WEIGHT_DECAY = 1e-2

    # Scheduler (OneCycleLR)
    PCT_START = 0.3  # 30% of training for warm-up
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 1000.0

    # Early Stopping
    PATIENCE = 5  # Stop if validation AUC doesn't improve for 5 epochs

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup on import
Config.setup()
