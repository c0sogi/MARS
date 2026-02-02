import os
import torch


class Config:
    """
    Configuration for the Augmented-Stem Dense-Feedback Recurrent Network (AS-DFRN).
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea/experiment
    WORKING_DIR = "./working/idea_73"

    # Ensure the working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache File Names (Keys)
    # Explicitly versioned to ensure clean feature generation
    TRAIN_CACHE_KEY = "train_data_as_dfrn_v1.npz"
    VAL_CACHE_KEY = "val_data_as_dfrn_v1.npz"
    TEST_CACHE_KEY = "test_data_as_dfrn_v1.npz"

    # Full Cache Paths
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, TRAIN_CACHE_KEY)
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, VAL_CACHE_KEY)
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, TEST_CACHE_KEY)

    # Output Paths
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LENGTH = 107
    SCORABLE_LENGTH = 68
    NUM_TARGETS = 5

    # Indices corresponding to scored columns:
    # reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    # Note: deg_pH10 is index 2, deg_50C is index 4 (unscored)
    SCORED_INDICES = [0, 1, 3]

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # General
    DIM = 64  # Latent dimension (Z) and RNN hidden size
    DROPOUT = 0.1

    # Backbone (Post-Activation Dense Dilated TCN)
    GROWTH_RATE = 64  # High capacity for the static backbone
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4, 8, 16, 32]  # Exponential dilation schedule

    # Feedback Module (Dense-Feedback TCN)
    FEEDBACK_GROWTH_RATE = 16  # Lightweight to prevent feedback dominance

    # =========================================================================
    # Training Settings
    # =========================================================================
    BATCH_SIZE = 16  # Strictly set to 16 for gradient frequency
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Algorithm Flags
    # =========================================================================
    # Mask unscored channels (indices 2 and 4) in the feedback loop
    CHANNEL_MASKING = True

    # Train on full 107 length with tail targets set to 0.0 to anchor RNN
    # If False, loss would be masked to first 68 positions only.
    BOUNDARY_ANCHORING = True
