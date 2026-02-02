import os
import torch


class Config:
    """
    Configuration for the Asymmetric Bottleneck Dense-Context Network.
    Acts as the single source of truth for paths, hyperparameters, and settings.
    """

    # --------------------------------------------------------------------------
    # Directory & File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_23"

    # Create working directory if it doesn't exist
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata CSVs
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Files (Unique keys for this idea to ensure fresh feature generation)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_asymmetric_dense_v1.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_asymmetric_dense_v1.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_asymmetric_dense_v1.npz")

    # --------------------------------------------------------------------------
    # Data Parameters
    # --------------------------------------------------------------------------
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Input Feature Dimensions
    # Sequence (4: A,G,C,U)
    # Structure (3: .,(,))
    # Loop Type (7: S,M,I,B,H,E,X)
    # Partner Identity (5: A,G,C,U, None) -> Explicit input feature
    NUM_NODE_FEATURES = 4 + 3 + 7 + 5  # Total: 19

    # Target Columns
    # The full list of 5 targets provided in training data
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    NUM_TARGETS = 5

    # Scored Targets (Subset used for Metric and Loss)
    # We only compute loss on: reactivity, deg_Mg_pH10, deg_Mg_50C
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Indices of the scored targets within the 5-column target vector
    # reactivity(0), deg_Mg_pH10(1), deg_pH10(2), deg_Mg_50C(3), deg_50C(4)
    # Indices: 0, 1, 3
    SCORED_TARGET_INDICES = [0, 1, 3]

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    # Backbone: Dense Dilated TCN
    DILATIONS = [1, 2, 4, 8, 16, 32]
    CHANNEL_WIDTH = 64  # Growth rate / Hidden channels in TCN blocks
    KERNEL_SIZE = 3
    DROPOUT = 0.1

    # Asymmetric Bottleneck Dimensions
    LOCAL_DIM = 96  # Stream 1: High-fidelity local representation
    MESSAGE_DIM = 32  # Stream 2: Compressed message for interaction

    # Global Aggregation
    GRU_HIDDEN_DIM = 64  # BiGRU hidden size (Output dim will be 2 * 64 = 128)

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    PATIENCE = 10  # Early stopping patience

    # Hardware & Reproducibility
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2
    SEED = 42

    # Debugging
    DEBUG = False  # If True, runs on a small subset
    SUBSET_SIZE = 100  # Size of subset if DEBUG is True
