import os
import torch


class Config:
    """
    Configuration class for the Anchored Dense-Feedback Recurrent Network (ADF-RN).
    """

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_77"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache filenames
    TRAIN_CACHE_FILE = "train_data_adf_rn_v1.npz"
    VAL_CACHE_FILE = "val_data_adf_rn_v1.npz"
    TEST_CACHE_FILE = "test_data_adf_rn_v1.npz"

    # =========================================================================
    # Data Settings
    # =========================================================================
    SEQ_LEN = 107
    SCORED_LEN = 68

    # Columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # The subset of columns used for the competition metric
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # Columns to mask in the feedback loop (unscored in metric)
    UNSCORED_COLS = ["deg_pH10", "deg_50C"]

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Backbone (Dense Dilated TCN)
    DILATIONS = [1, 2, 4, 8, 16, 32]
    BACKBONE_GROWTH_RATE = 64
    LATENT_DIM = 64
    KERNEL_SIZE = 3

    # Feedback Module
    FEEDBACK_GROWTH_RATE = 16
    FEEDBACK_OUT_DIM = 32

    # RNN Aggregator
    RNN_HIDDEN_DIM = 64
    BIDIRECTIONAL = True

    # General
    DROPOUT = 0.1
    NUM_TARGETS = 5

    # Input Features
    # Sequence (4) + Structure (3) + LoopType (7) + Partner Identity (4)
    INPUT_CHANNELS = 4 + 3 + 7 + 4

    # =========================================================================
    # Training Settings
    # =========================================================================
    BATCH_SIZE = 16  # Strictly set to 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 25
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss Weights
    PASS1_LOSS_WEIGHT = 0.5
    PASS2_LOSS_WEIGHT = 1.0

    # =========================================================================
    # Debugging / Development
    # =========================================================================
    # If set to an integer, only load this many samples for training/val
    SUBSET_SIZE = None
    DEBUG = False

    @classmethod
    def get_cache_path(cls, filename):
        return os.path.join(cls.WORKING_DIR, filename)
