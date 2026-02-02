import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    PROJECT_NAME = "RNA_Degradation_Prediction"
    EXPERIMENT_NAME = "idea_24"  # Token-Adaptive Wide-Stream Residual BiGRU
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2  # Adjust based on CPU availability

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output Directories
    WORKING_DIR = f"./working/{EXPERIMENT_NAME}"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Create directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Output Files
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68  # Number of positions with ground truth

    # Token Mappings
    TOKEN_MAP = {"A": 0, "G": 1, "C": 2, "U": 3}
    LOOP_TYPE_MAP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    VOCAB_SIZE = len(TOKEN_MAP)
    LOOP_VOCAB_SIZE = len(LOOP_TYPE_MAP)

    # Targets used for training and scoring
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Columns to ignore during training (unscored)
    IGNORE_COLS = ["deg_pH10", "deg_50C"]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Embedding dimensions
    EMBED_DIM = 100  # Dimension for nucleotide embeddings
    LOOP_EMBED_DIM = 64  # Dimension for loop type embeddings
    DIST_EMBED_DIM = 64  # Dimension for distance embeddings (sinusoidal projection)

    # Main Recurrent Architecture
    # The Wide-Stream width W.
    # Input projection will map concatenated embeddings to this dim.
    HIDDEN_DIM = 384

    # Number of Residual BiGRU blocks
    NUM_LAYERS = 6

    # Dropout rates
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0

    # Scheduler
    T_MAX = EPOCHS  # For Cosine Annealing
    ETA_MIN = 1e-6

    # Validation
    PRINT_FREQ = 10  # Print training metrics every N batches

    def __init__(self):
        # Ensure reproducibility upon instantiation
        self.set_seed()

    @classmethod
    def set_seed(cls):
        import random
        import numpy as np

        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
