import os
import torch


class Config:
    # =========================================================================
    # General Configuration
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Data (Metadata Parquet files)
    METADATA_DIR = "./metadata"
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.parquet")
    VAL_FILE = os.path.join(METADATA_DIR, "val.parquet")
    TEST_FILE = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_FILE = "./input/sample_submission.csv"

    # Working Directory for Idea 26
    WORKING_DIR = "./working/idea_26"

    # Sub-directories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Specific Output Files
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specifics
    # =========================================================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Target Columns to be predicted and scored
    # Note: deg_pH10 and deg_50C are ignored during training as per strategy
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Vocabularies
    # Nucleotides: A, G, C, U
    TOKEN_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
    VOCAB_SIZE = len(TOKEN_MAP)

    # Loop Types: S, M, I, B, H, E, X
    LOOP_TYPE_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
    LOOP_VOCAB_SIZE = len(LOOP_TYPE_MAP)

    # =========================================================================
    # Model Architecture Hyperparameters
    # =========================================================================
    # Embedding Dimensions
    EMBED_DIM_SEQ = 64  # Dimension for Atomic Sequence Embeddings
    EMBED_DIM_LOOP = 32  # Dimension for Predicted Loop Type Embeddings
    EMBED_DIM_POS = 32  # Dimension for Absolute Positional Embeddings

    # Wide-Stream Backbone
    HIDDEN_SIZE = 384  # Residual Stream Width (W)
    NUM_LAYERS = 6  # Number of Residual BiGRU Blocks

    # Regularization
    DROPOUT = 0.1  # General Dropout
    STRUCTURAL_DROPOUT_PROB = (
        0.15  # Probability to drop structural edges (pair distance -> 0)
    )

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Loss Function
    # Using MSE (L2) on the first 68 positions
    LOSS_FN = "mse"

    # =========================================================================
    # Debug / Development
    # =========================================================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SUBSET_SIZE = 100  # Number of samples to use in debug mode

    @classmethod
    def setup(cls):
        """
        Creates the necessary directory structure for the experiment.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories initialized at {cls.WORKING_DIR}")
