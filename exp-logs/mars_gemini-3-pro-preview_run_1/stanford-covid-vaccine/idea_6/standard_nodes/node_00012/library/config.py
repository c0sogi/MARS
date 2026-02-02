import os
import torch
import random
import numpy as np


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Config:
    # =========================================================================
    # General Configuration
    # =========================================================================
    PROJECT_NAME = "RNA_Degradation_Hybrid_ResNet_BiGRU"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SUBSET_SIZE = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of subprocesses for data loading

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    # Root directories
    ROOT_DIR = "."
    INPUT_DIR = os.path.join(ROOT_DIR, "input")
    METADATA_DIR = os.path.join(ROOT_DIR, "metadata")
    WORKING_DIR = os.path.join(ROOT_DIR, "working", "idea_6")
    SUBMISSION_DIR = os.path.join(ROOT_DIR, "submission")

    # Input Data Files (Parquet Metadata)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Sample Submission (for format reference)
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    CACHE_DIR = WORKING_DIR  # Directory to store cached processed tensors
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107  # Total length of RNA sequence
    SCORED_LEN = 68  # Number of positions scored in the competition

    # Target Columns used for scoring
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Vocabulary Mappings
    # Sequence: A, G, U, C
    NUCLEOTIDE_MAP = {"A": 0, "G": 1, "U": 2, "C": 3}
    VOCAB_SIZE_SEQ = len(NUCLEOTIDE_MAP)

    # Structure: (, ), .
    STRUCTURE_MAP = {"(": 0, ")": 1, ".": 2}
    VOCAB_SIZE_STRUCT = len(STRUCTURE_MAP)

    # Predicted Loop Type: B, E, H, I, M, S, X
    LOOP_MAP = {"B": 0, "E": 1, "H": 2, "I": 3, "M": 4, "S": 5, "X": 6}
    VOCAB_SIZE_LOOP = len(LOOP_MAP)

    # =========================================================================
    # Model Architecture Hyperparameters (Hybrid ResNet-BiGRU)
    # =========================================================================
    # Embedding
    EMBED_DIM = 64  # Dimension for each input type (Seq, Struct, Loop)

    # 1D-ResNet (Local Feature Encoder)
    RESNET_CHANNELS = 128  # Number of channels in ResNet blocks
    RESNET_BLOCKS = 4  # Number of Residual blocks
    RESNET_KERNEL_SIZE = 3  # Kernel size for convolutions

    # Bi-GRU (Global Context Aggregator)
    GRU_HIDDEN_SIZE = 256  # Hidden state size for GRU
    GRU_LAYERS = 2  # Number of stacked GRU layers
    BIDIRECTIONAL = True  # Use bidirectional GRU

    # Regularization
    DROPOUT = 0.3  # Dropout probability

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # For AdamW
    EPOCHS = 20

    # Early Stopping
    PATIENCE = 5  # Stop if validation loss doesn't improve for N epochs

    # Loss Function
    HUBER_DELTA = 1.0  # Delta parameter for Huber Loss (Robustness to noise)

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR

    @classmethod
    def setup_workspace(cls):
        """
        Creates necessary directories for outputs and cache.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Workspace setup complete. Working dir: {cls.WORKING_DIR}")


# Initialize environment immediately upon import
set_seed(Config.SEED)
Config.setup_workspace()
