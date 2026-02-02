import os
import torch


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Implements the settings for the Spectral-Topological Wide-Stream Residual BiGRU strategy.
    """

    # =========================================================================
    # Directories and File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_49"

    # Input Data Files (Parquet format from metadata generation)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Submission Files
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Targets to be trained on (Scored columns only)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Vocabularies
    # Sequence: A, G, C, U
    VOCAB_SIZE_SEQ = 4
    # Loop Type: S, M, I, B, H, E, X
    VOCAB_SIZE_LOOP = 7

    # =========================================================================
    # Model Architecture Hyperparameters
    # =========================================================================
    # Strategy: Spectral-Topological Wide-Stream Residual BiGRU

    # 1. Heterogeneous Feature Embeddings
    EMBED_DIM_SEQ = 128  # Atomic sequence embedding
    EMBED_DIM_LOOP = 64  # Predicted loop type embedding
    EMBED_DIM_PAIR = 64  # Signed Sinusoidal Pairing Distance embedding

    # 2. Laplacian Positional Encodings (LPE)
    LPE_DIM = 8  # Number of eigenvectors to extract (k)
    LPE_EMBED_DIM = 32  # Dimension to project the k eigenvectors into

    # 3. Backbone (BiGRU)
    # Explicitly using Width 384 and 6 Layers as per strategy
    HIDDEN_DIM = 384  # Residual stream width
    NUM_LAYERS = 6  # Number of Wide-Stream Residual Blocks
    DROPOUT = 0.2  # Inter-layer dropout (strictly no stem dropout)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    EPOCHS = 20
    BATCH_SIZE = 32

    # Optimization
    LEARNING_RATE = 1e-3  # Standard for AdamW
    WEIGHT_DECAY = 1e-4  # Low weight decay to preserve recurrent signals
    GRAD_CLIP = 1.0  # Gradient clipping norm

    # Loss
    LOSS_FN = "MSE"  # Strictly MSE (L2)

    # =========================================================================
    # System & Hardware
    # =========================================================================
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging / Development
    # Set to None to use full dataset, or an integer (e.g., 100) for quick testing
    MAX_DEBUG_SAMPLES = None

    @classmethod
    def initialize(cls):
        """
        Sets up the working directory and ensures reproducibility.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        print(f"Configuration initialized. Working directory: {cls.WORKING_DIR}")
        print(f"Device: {cls.DEVICE}")


# Initialize environment immediately upon import
Config.initialize()
