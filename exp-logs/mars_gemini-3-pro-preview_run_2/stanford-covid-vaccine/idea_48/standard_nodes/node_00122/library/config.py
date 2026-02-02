import os
import torch


class Config:
    """
    Configuration for the Embedded-Input Pure-Feedback Network (EI-PFN).
    Centralizes hyperparameters, file paths, and data settings.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_48"

    # Create working directory if it doesn't exist
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache File Paths (using .npz for efficient storage)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_ei_pfn_v1.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_ei_pfn_v1.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_ei_pfn_v1.npz")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Dimensions & Features
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Feature Counts for One-Hot Encoding
    NUM_SEQUENCE_TYPES = 4  # A, G, U, C
    NUM_STRUCTURE_TYPES = 3  # (, ), .
    NUM_LOOP_TYPES = 7  # S, M, I, B, H, E, X
    NUM_PARTNER_TYPES = 5  # A, G, U, C, None (Partner Identity)

    # Total Input Channels = Sum of feature types
    INPUT_CHANNELS = (
        NUM_SEQUENCE_TYPES + NUM_STRUCTURE_TYPES + NUM_LOOP_TYPES + NUM_PARTNER_TYPES
    )

    # =========================================================================
    # Model Architecture Hyperparameters
    # =========================================================================
    # Input Embedding Stem
    NUM_INIT_FEATURES = 64  # Project inputs to this dim before backbone

    # Main Backbone (Static Dense Dilated TCN)
    GROWTH_RATE = 64
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4, 8, 16, 32]
    DROPOUT = 0.1
    LATENT_DIM = 64  # Dimension of Z (static features)

    # Pure-Feedback Module
    FEEDBACK_GROWTH_RATE = 16
    FEEDBACK_OUT_CHANNELS = 32  # Dimension of E_fb

    # Aggregation (Interaction + RNN)
    RNN_HIDDEN_SIZE = 64
    RNN_LAYERS = 1
    RNN_BIDIRECTIONAL = True

    # =========================================================================
    # Targets and Scoring
    # =========================================================================
    # All targets provided in training data
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Targets used for scoring (and strict masking in feedback)
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    NUM_TARGETS = 5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    PATIENCE = 10  # Early stopping patience
    NUM_WORKERS = 4

    # Loss Weights
    LOSS_WEIGHT_PASS1 = 0.5  # Weight for initial prediction (no feedback)
    LOSS_WEIGHT_PASS2 = 1.0  # Weight for refined prediction (with feedback)

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
