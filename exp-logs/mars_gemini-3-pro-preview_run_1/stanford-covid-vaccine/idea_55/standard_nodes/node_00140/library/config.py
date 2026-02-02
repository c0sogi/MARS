import os
import torch


class Config:
    """
    Configuration for the Scaled-Residual Wide-Stream BiGRU strategy.
    Encapsulates paths, data settings, model architecture, and training hyperparameters.
    """

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    WORKING_DIR = "./working"
    # Cache directory for idea_55 specific processing
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_55")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Targets to be predicted and scored
    # Note: Training is restricted to these 3 columns (Node 00019)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Vocab sizes for embeddings
    VOCAB_SIZE_SEQ = 4  # A, G, C, U
    VOCAB_SIZE_LOOP = 7  # B, E, H, I, M, S, X
    VOCAB_SIZE_STRUCT = 3  # (, ), .

    # Debugging
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100

    # =========================================================================
    # Model Architecture: Scaled-Residual Wide-Stream BiGRU
    # =========================================================================
    # Embedding Dimensions (Heterogeneous Feature Embedding - Node 00099)
    EMBED_DIM_SEQ = 128
    EMBED_DIM_LOOP = 64
    EMBED_DIM_PAIR = 64  # For Signed Sinusoidal Pairing Distance

    # Backbone Dimensions
    # Explicitly target high capacity width 512 (Node 00022)
    HIDDEN_DIM = 512
    NUM_LAYERS = 6

    # Regularization & Stability
    DROPOUT = 0.2  # Inter-layer dropout (Node 00076)
    USE_STEM_DROPOUT = False  # No dropout after initial projection (Node 00109)

    # Residual Scaling
    USE_LAYER_SCALE = True
    INIT_LAYER_SCALE = 1.0  # Initialize at 1.0 to preserve signal flow (Node 00084)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Optimization
    OPTIMIZER = "AdamW"
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Low weight decay to preserve recurrent signal (Node 00070)
    GRAD_CLIP = 1.0  # Critical stabilizer for 512-width backbone (Node 00097)

    # Batching & Scheduling
    BATCH_SIZE = 32  # Strictly 32 (Node 00131)
    EPOCHS = 20
    SCHEDULER = "CosineAnnealing"

    # Loss
    LOSS_FN = "MSE"  # Mean Squared Error (Node 00012)
    MASK_LOSS = True  # Calculate loss only on first 68 positions (Node 00002)

    # Hardware
    NUM_WORKERS = 4
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist."""
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
