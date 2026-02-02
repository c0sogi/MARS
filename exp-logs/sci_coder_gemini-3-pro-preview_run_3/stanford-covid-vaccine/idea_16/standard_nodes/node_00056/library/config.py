import os
import torch


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Implements the settings for the Interleaved Gated-Structure BiGRU strategy.
    """

    # =========================================================================
    # File Paths
    # =========================================================================
    # Metadata directories (Input)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Working directories (Output)
    # Using idea_16 as specified in the strategy
    WORKING_DIR = "./working/idea_16"

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68
    NUM_TARGETS = 5

    # Target Columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Columns used for the competition metric
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Feature Mappings (One-Hot Encoding)
    # Sequence: 4 channels
    TOKEN2INT_SEQ = {x: i for i, x in enumerate("AGCU")}
    # Structure: 3 channels
    TOKEN2INT_STRUCT = {x: i for i, x in enumerate("().")}
    # Predicted Loop Type: 7 channels
    TOKEN2INT_LOOP = {x: i for i, x in enumerate("SMIBHEX")}

    # Total Input Channels = 4 + 3 + 7 = 14
    INPUT_CHANNELS = len(TOKEN2INT_SEQ) + len(TOKEN2INT_STRUCT) + len(TOKEN2INT_LOOP)

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Architecture: Interleaved Gated-Structure BiGRU
    HIDDEN_DIM = 384
    NUM_LAYERS = 3  # Number of BiGRU + Injection blocks
    DROPOUT = 0.1

    # Convolutional Stem
    CONV_FILTERS = 256
    CONV_KERNEL_SIZE = 3

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 20

    # Optimization
    GRAD_CLIP_NORM = 1.0  # Mandatory for stability with deep RNNs
    WEIGHT_DECAY = 1e-4  # For AdamW
    SCHEDULER_T_MAX = EPOCHS  # For CosineAnnealingLR

    # =========================================================================
    # System & Reproducibility
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Debugging & Development
    # =========================================================================
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100  # Size of subset when DEBUG is True

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
