import os
import torch


class Config:
    # --------------------------------------------------------------------------
    # Directory and File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for Idea 52
    WORKING_DIR = "./working/idea_52"
    SUBMISSION_DIR = "./submission"

    # Input Files (Using generated Parquet Metadata)
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.parquet")
    VAL_FILE = os.path.join(METADATA_DIR, "val.parquet")
    TEST_FILE = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    SEQ_LEN = 107
    PRED_LEN = 68

    # Vocabulary Mappings
    # Sequence: A, G, C, U (Atomic tokens)
    TOKEN2ID_SEQ = {"A": 0, "G": 1, "C": 2, "U": 3}
    ID2TOKEN_SEQ = {v: k for k, v in TOKEN2ID_SEQ.items()}
    VOCAB_SIZE_SEQ = len(TOKEN2ID_SEQ)

    # Loop Types: S, M, I, B, H, E, X
    TOKEN2ID_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
    ID2TOKEN_LOOP = {v: k for k, v in TOKEN2ID_LOOP.items()}
    VOCAB_SIZE_LOOP = len(TOKEN2ID_LOOP)

    # Targets to train on (Target Filtering)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_CLASSES = len(TARGET_COLS)

    # --------------------------------------------------------------------------
    # Model Architecture (Position-Aware Proportional Wide-Stream BiGRU)
    # --------------------------------------------------------------------------
    # Proportional Input Embeddings
    EMB_SEQ_DIM = 128  # Atomic Sequence
    EMB_LOOP_DIM = 64  # Predicted Loop Type
    EMB_PAIR_DIM = 64  # Signed Sinusoidal Pairing Distance
    EMB_POS_DIM = 32  # Absolute Positional Encoding

    # Total input dimension for the BiGRU stem (Concatenated)
    # 128 + 64 + 64 + 32 = 288
    INPUT_DIM = EMB_SEQ_DIM + EMB_LOOP_DIM + EMB_PAIR_DIM + EMB_POS_DIM

    # Backbone Hyperparameters
    HIDDEN_SIZE = 384  # Wide-Stream capacity (strictly 384, not 512)
    NUM_LAYERS = 6  # Residual Blocks
    DROPOUT = 0.2  # Inter-layer dropout (strictly applied)

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32  # Strictly 32
    EPOCHS = 20  # Fixed number of epochs for Cosine Annealing
    LEARNING_RATE = 1e-3  # Standard AdamW start
    WEIGHT_DECAY = 1e-4  # Low weight decay to preserve recurrent signals
    MAX_GRAD_NORM = 1.0  # Gradient clipping for stability

    # --------------------------------------------------------------------------
    # System / Misc
    # --------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, debug=False):
        """
        Initialize configuration.
        Args:
            debug (bool): If True, adjusts parameters for quick debugging runs.
        """
        self.debug = debug

        # Ensure working directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        if self.debug:
            self.EPOCHS = 2
            print(f"Debug mode enabled. Epochs set to {self.EPOCHS}.")
