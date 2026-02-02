import os
import torch


class Config:
    """
    Configuration for the Topologically-Augmented Stabilized Wide-Stream BiGRU strategy.
    """

    # --------------------------------------------------------------------------
    # General & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Debugging flags (can be used by training scripts to limit dataset size)
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100

    # --------------------------------------------------------------------------
    # File System Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Directory for caching processed features (RWPE, etc.)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_50")

    # Directory and path for the final submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Specifications
    # --------------------------------------------------------------------------
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Ground truth columns to train on (filtering out deg_pH10 and deg_50C)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Vocabulary Mappings
    # Sequence: A, G, U, C
    TOKEN2ID_SEQ = {"A": 0, "G": 1, "U": 2, "C": 3}
    VOCAB_SIZE_SEQ = 4

    # Structure: ., (, )
    TOKEN2ID_STRUCT = {".": 0, "(": 1, ")": 2}
    VOCAB_SIZE_STRUCT = 3

    # Predicted Loop Type: B, E, H, I, M, S, X
    TOKEN2ID_LOOP = {"B": 0, "E": 1, "H": 2, "I": 3, "M": 4, "S": 5, "X": 6}
    VOCAB_SIZE_LOOP = 7

    # --------------------------------------------------------------------------
    # Model Architecture (Topologically-Augmented Stabilized Wide-Stream BiGRU)
    # --------------------------------------------------------------------------
    # Feature Embeddings
    EMBED_DIM_SEQ = 128  # High-dim for atomic sequence
    EMBED_DIM_LOOP = 64  # Moderate-dim for local context
    EMBED_DIM_PAIR = 64  # Moderate-dim for pair distance

    # Random Walk Positional Encoding (RWPE)
    # Steps k for T^k diagonal elements
    RWPE_STEPS = [1, 2, 4, 8, 16]

    # Recurrent Backbone
    HIDDEN_DIM = 512  # Wide-Stream capacity
    NUM_LAYERS = 6  # Deep encoder
    DROPOUT = 0.2  # Applied between layers (not on stem output)

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 64
    NUM_EPOCHS = 20

    # Optimizer: AdamW
    LR = 1e-3
    WEIGHT_DECAY = 1e-4  # Low weight decay to preserve recurrent signal

    # Stabilization
    GRAD_CLIP = 1.0  # Critical for width 512 convergence

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for caching and submission.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
