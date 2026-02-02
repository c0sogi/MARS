import os
import torch


class Config:
    # =========================================================================
    # Directories and File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_22"
    SUBMISSION_DIR = "./submission"

    # Input Files (Parquet format from metadata generation)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Submission Templates
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Vocabularies
    # Atomic Nucleotide Tokens
    TOKEN_TO_ID = {"A": 0, "G": 1, "C": 2, "U": 3}
    VOCAB_SIZE = len(TOKEN_TO_ID)

    # Predicted Loop Types
    LOOP_TO_ID = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
    LOOP_VOCAB_SIZE = len(LOOP_TO_ID)

    # Targets
    # Columns used for calculating Loss (Ground truth available)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Columns required in the final submission file
    SUBMISSION_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # =========================================================================
    # Model Architecture (Dense-Aggregated Wide-Stream Residual BiGRU)
    # =========================================================================
    # Residual Stream Width (W)
    HIDDEN_DIM = 384

    # Depth
    N_LAYERS = 6

    # Regularization
    DROPOUT = 0.1

    # Input Embedding Dimensions
    NUC_EMBED_DIM = 32  # For Atomic Sequence (Cite solution_lesson_node_00049)
    LOOP_EMBED_DIM = 32  # For Loop Type
    POS_EMBED_DIM = 32  # For Sinusoidal Pairing Distance

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.01
    EPOCHS = 25
    MAX_GRAD_NORM = 1.0

    # Scheduler: ReduceLROnPlateau
    SCHEDULER_PATIENCE = 4
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    # =========================================================================
    # System & Reproducibility
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Debugging / Development
    # =========================================================================
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100  # Number of samples to use if DEBUG is True

    @classmethod
    def setup(cls):
        """Ensures working and submission directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
