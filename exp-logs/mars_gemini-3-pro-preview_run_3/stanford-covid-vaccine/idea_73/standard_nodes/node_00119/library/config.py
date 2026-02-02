import os
import torch
import random
import numpy as np


class Hyperparameters:
    """
    Central configuration for the RNA Degradation Prediction task.
    Implements the 'High-Capacity Full-Rank Synthesis' strategy parameters.
    """

    # --------------------------------------------------------------------------
    # System & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Optimized for the 12 vCPU environment

    # --------------------------------------------------------------------------
    # File Paths & Directories
    # --------------------------------------------------------------------------
    # Input Metadata (Pre-generated Parquet files)
    TRAIN_DATA_PATH = "./metadata/train.parquet"
    VAL_DATA_PATH = "./metadata/val.parquet"
    TEST_DATA_PATH = "./metadata/test.parquet"

    # Sample Submission
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output & Cache Directories
    # Strategy-specific working directory
    WORKING_DIR = "./working/idea_73/"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODELS_DIR = os.path.join(WORKING_DIR, "models")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure critical directories exist immediately upon config load
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # --------------------------------------------------------------------------
    # Data Specifications
    # --------------------------------------------------------------------------
    SEQ_LENGTH = 107
    SEQ_SCORED = 68
    NUM_TARGETS = 5  # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    # Input Features:
    # 4 (Nucleotides: A, G, C, U)
    # + 3 (Structure: ., (, ))
    # + 7 (Loop Type: S, M, I, B, H, E, X)
    INPUT_DIM = 14

    # --------------------------------------------------------------------------
    # Model Architecture: High-Capacity Full-Rank GLU-Decoupled BiGRU
    # --------------------------------------------------------------------------
    # Backbone Configuration
    # Hidden dimension per direction. Total hidden size will be 2 * 384 = 768.
    HIDDEN_DIM = 384
    BIDIRECTIONAL = True
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # Convolutional Stem Configuration
    STEM_KERNEL_SIZE = 3
    STEM_FILTERS = 256

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    # Optimization
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Stabilization
    GRADIENT_CLIP = 1.0  # Strict clipping to prevent exploding gradients in deep RNN

    # Scheduling & Duration
    NUM_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Debugging / Development
    # Fraction of dataset to use (1.0 = use all data).
    # Can be reduced (e.g., 0.1) for rapid debugging.
    DATA_SUBSET_FRACTION = 1.0

    @staticmethod
    def set_seed(seed=42):
        """
        Sets fixed random seeds for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior in cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
