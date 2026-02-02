import os
import torch
import numpy as np
import random


class Config:
    # ==============================
    # File Paths
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_22"

    # Metadata Paths (Parquet files)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Cache Paths
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_cache.npy")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_cache.npy")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_cache.npy")

    # ==============================
    # Data Parameters
    # ==============================
    SEQ_LEN = 107
    PRED_LEN = 68
    # Targets: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    NUM_CLASSES = 5

    # Input Features:
    # 4 (Sequence: A,G,C,U) + 3 (Structure: .,(,)) + 7 (Loop: S,M,I,B,H,E,X)
    INPUT_CHANNELS = 14
    USE_ONE_HOT = True

    # ==============================
    # Model Architecture (Idea 22)
    # ==============================
    # Convolutional Stem
    CONV_FILTERS = 256
    KERNEL_SIZE = 3

    # Refinement Backbone (BiGRU + Channel Gating)
    HIDDEN_DIM = 384  # Strictly 384 as per lessons
    NUM_LAYERS = 3
    DROPOUT = 0.1

    # ==============================
    # Training Hyperparameters
    # ==============================
    SEED = 2024
    BATCH_SIZE = 32  # Adjusted for 384 dim + BiGRU overhead
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0  # Mandatory for stability

    # Compute
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, debug=False, epochs=None, batch_size=None, **kwargs):
        """
        Initialize configuration with optional overrides.

        Args:
            debug (bool): If True, enables debug mode (fewer epochs, subset of data).
            epochs (int): Override default number of epochs.
            batch_size (int): Override default batch size.
            **kwargs: Additional overrides for config attributes.
        """
        # Create working directory
        os.makedirs(self.WORKING_DIR, exist_ok=True)

        # Apply overrides
        if epochs is not None:
            self.EPOCHS = epochs

        if batch_size is not None:
            self.BATCH_SIZE = batch_size

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # Debug mode settings
        self.debug = debug
        if self.debug:
            self.EPOCHS = 2
            self.BATCH_SIZE = 16
            print(
                f"Debug mode enabled: EPOCHS={self.EPOCHS}, BATCH_SIZE={self.BATCH_SIZE}"
            )

    @staticmethod
    def set_seed(seed=None):
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
        """
        if seed is None:
            seed = Config.SEED

        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {seed}")
