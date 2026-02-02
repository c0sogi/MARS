import os
import torch


class Config:
    """
    Configuration for the Sequence-Gated Wide-Stream Residual BiGRU model.
    Includes paths, hyperparameters, and data settings.
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and saving checkpoints
    # Using 'idea_62' to isolate this experiment's artifacts
    WORKING_DIR = "./working/idea_62"

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Data Files (using pre-generated Parquet metadata)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Feature Columns
    ID_COL = "id"
    SEQUENCE_COL = "sequence"
    STRUCTURE_COL = "structure"
    LOOP_TYPE_COL = "predicted_loop_type"

    # Targets
    # We train ONLY on the 3 scored columns to reduce noise
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # All target columns required for the submission format
    ALL_TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # =========================================================================
    # Model Hyperparameters
    # Strategy: Sequence-Gated Wide-Stream Residual BiGRU
    # =========================================================================
    HIDDEN_DIM = 384  # Wide-Stream width (Residual Stream)

    # Embedding Dimensions (Heterogeneous Feature Embedding)
    EMBED_DIM_SEQ = 128  # Atomic Sequence Identity (A, G, C, U)
    EMBED_DIM_LOOP = 64  # Predicted Loop Type
    EMBED_DIM_DIST = 64  # Signed Sinusoidal Pairing Distance

    # Architecture Depth
    NUM_LAYERS = 6  # Number of Sequence-Gated Residual Blocks

    # Regularization
    DROPOUT = 0.2  # Inter-Layer Dropout

    # Output
    OUTPUT_DIM = 3  # Corresponds to len(TARGET_COLS)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32  # Strictly 32
    LEARNING_RATE = 1e-3  # Standard starting LR for AdamW
    WEIGHT_DECAY = 1e-4  # Low weight decay to preserve recurrent signal
    CLIP_NORM = 1.0  # Gradient clipping to stabilize BiGRU
    EPOCHS = 20  # Fixed number of epochs for Cosine Annealing

    # =========================================================================
    # Runtime & Reproducibility
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging / Development
    # Toggle DEBUG to True to run on a small subset of data
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100

    @staticmethod
    def create_dirs():
        """Creates necessary working and submission directories if they don't exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
