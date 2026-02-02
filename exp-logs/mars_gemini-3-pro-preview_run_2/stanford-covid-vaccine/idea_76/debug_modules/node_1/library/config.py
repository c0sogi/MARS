import os
import torch


class Config:
    """
    Configuration for the Anchored High-Capacity Hybrid-Input Dense Network (AHC-HIDN).
    """

    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_76"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # =========================================================================
    # Data Configuration
    # =========================================================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Cache file name for this specific strategy
    CACHE_NAME = "train_data_ahc_hidn_v1.npz"

    # Target Columns in the standard order
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Indices of columns that are actually scored in the competition
    # 0: reactivity, 1: deg_Mg_pH10, 3: deg_Mg_50C
    SCORED_COLS_INDICES = [0, 1, 3]

    # Indices of columns to mask out during feedback (unscored targets)
    # We mask deg_pH10 (2) and deg_50C (4) to prevent unsupervised noise injection
    UNSCORED_COLS_INDICES = [2, 4]

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Backbone: High-Capacity Dense Dilated TCN
    GROWTH_RATE = 64
    LATENT_DIM = 64
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4, 8, 16, 32]
    DROPOUT = 0.1

    # Feedback Module: Lightweight Dense TCN
    FEEDBACK_GROWTH_RATE = 16
    FEEDBACK_DIM = 32

    # Global Aggregation: Bidirectional GRU
    RNN_HIDDEN_DIM = 64  # Compact hidden size

    # Input Features
    # 4 bases + 1 paired_base_identity (4) + 1 loop_type (7) + 1 structure (3) = ~19 raw channels
    # But we use one-hot encodings, so dimensions are handled dynamically in dataset.

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 16  # Strictly set to 16 for gradient frequency
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    NUM_WORKERS = 2
    SEED = 42

    # Loss Strategy
    USE_ANCHORED_LOSS = True  # Calculate loss over full 0-107 sequence
    AUX_LOSS_WEIGHT = 0.5  # Weight for the first pass prediction (Pass 1)

    # Recycling
    RECYCLING_STEPS = 2  # Pass 1 (Zero Feedback) + Pass 2 (Detached Feedback)

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Debugging / Development
    # =========================================================================
    # Set to a small integer (e.g., 100) to train on a subset for debugging.
    # Set to None to train on the full dataset.
    SUBSET_SIZE = None
