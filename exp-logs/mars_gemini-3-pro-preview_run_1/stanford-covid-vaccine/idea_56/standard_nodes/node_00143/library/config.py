import os


class Config:
    """
    Configuration for the Channel-Scaled Wide-Stream Residual BiGRU model.
    Encapsulates all hyperparameters, constants, and file paths.
    """

    # --------------------------------------------------------------------------
    # General & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 4  # Optimized for the available 12 vCPUs

    # --------------------------------------------------------------------------
    # Data Specifications
    # --------------------------------------------------------------------------
    SEQ_LEN = 107
    PRED_LEN = 68

    # We train only on the 3 scored columns as per the strategy to reduce noise
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # --------------------------------------------------------------------------
    # Model Architecture (Channel-Scaled Wide-Stream Residual BiGRU)
    # --------------------------------------------------------------------------
    # Embedding Dimensions
    EMBED_DIM_SEQ = 128  # Atomic Sequence
    EMBED_DIM_LOOP = 64  # Predicted Loop Type
    EMBED_DIM_DIST = 64  # Signed Sinusoidal Pairing Distance

    # Backbone
    HIDDEN_DIM = 384  # Wide Stream Width (W)
    NUM_LAYERS = 6  # Number of Residual Blocks
    DROPOUT = 0.1  # Inter-layer Dropout (Cite Lesson 00112)

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32  # Strictly 32 to maintain gradient update frequency
    EPOCHS = 20  # Fixed epoch count for Cosine Annealing
    LR = 1e-3  # Standard starting learning rate for AdamW
    WEIGHT_DECAY = 1e-4  # Low weight decay to preserve recurrent signals
    MAX_GRAD_NORM = 1.0  # Gradient clipping to stabilize the recurrent backbone

    # --------------------------------------------------------------------------
    # File Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory specific to this idea/experiment
    WORKING_DIR = "./working/idea_56"
    SUBMISSION_DIR = "./submission"

    # Input Files (using Parquet metadata)
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.parquet")
    VAL_FILE = os.path.join(METADATA_DIR, "val.parquet")
    TEST_FILE = os.path.join(METADATA_DIR, "test.parquet")

    # Sample submission for formatting reference
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup()
