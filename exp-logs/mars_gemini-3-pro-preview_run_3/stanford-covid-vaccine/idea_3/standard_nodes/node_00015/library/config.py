import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration class for the Signal-Filtered Convolutional Transformer pipeline.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Paths
    # ==========================================
    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working Directory for Caching and Models
    WORKING_DIR = "./working/idea_3"

    # Metadata Files (Pre-split and generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")

    # Raw Input Files (for reference or submission format)
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Files (Numpy format for processed tensors)
    # We will store dictionaries or tuples in these .npy files
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data.npy")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data.npy")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data.npy")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "transformer_model.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    # Sequence dimensions
    SEQ_LEN = 107
    PRED_LEN = 68

    # Feature Dimensions
    # 4 Nucleotides (A, G, C, U) + 3 Structure (., (, )) + 7 Loop Types (S, M, I, B, H, E, X)
    # Total Input Channels = 4 + 3 + 7 = 14
    INPUT_CHANNELS = 14

    # Target Dimensions (reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C)
    NUM_TARGETS = 5
    SCORED_INDICES = [0, 1, 3]  # reactivity, deg_Mg_pH10, deg_Mg_50C

    # Filtering Strategy
    # If True, only training samples with SN_filter == 1 will be used.
    FILTER_SN = True

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Convolutional Tokenizer
    KERNEL_SIZE = 3  # Local context window

    # Transformer Encoder
    D_MODEL = 128  # Latent dimension size
    NHEAD = 4  # Number of attention heads
    NUM_LAYERS = 2  # Number of transformer encoder layers
    DIM_FEEDFORWARD = 512  # Dimension of the feedforward network model
    DROPOUT = 0.2  # Dropout rate

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 35
    EARLY_STOPPING_PATIENCE = 7

    # Scheduler parameters (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-5

    # ==========================================
    # Debugging / Development
    # ==========================================
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100  # Number of samples to use if DEBUG is True

    @staticmethod
    def setup_reproducibility(seed=SEED):
        """
        Sets random seeds for reproducibility across Python, NumPy, and PyTorch.
        """
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Deterministic algorithms can be slower, but ensure reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def create_dirs(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
