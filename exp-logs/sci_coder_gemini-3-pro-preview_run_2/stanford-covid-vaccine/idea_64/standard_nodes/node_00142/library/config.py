import os
import torch


class Config:
    """
    Configuration class for the Masked-Loss Global-Feedback Network (ML-GFN).
    Stores paths, hyperparameters, and constants.
    """

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Idea-specific working directory
    IDEA_NAME = "idea_64"
    IDEA_DIR = os.path.join(WORKING_DIR, IDEA_NAME)

    # Ensure necessary directories exist
    os.makedirs(IDEA_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Raw Input Paths (if needed)
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache File Paths (Explicit Cache Invalidation)
    # Using 'v1' suffix to ensure partner identity features are generated correctly
    CACHE_VERSION = "v1"
    TRAIN_CACHE = os.path.join(IDEA_DIR, f"train_data_ml_gfn_{CACHE_VERSION}.npz")
    VAL_CACHE = os.path.join(IDEA_DIR, f"val_data_ml_gfn_{CACHE_VERSION}.npz")
    TEST_CACHE = os.path.join(IDEA_DIR, f"test_data_ml_gfn_{CACHE_VERSION}.npz")

    # Output Paths
    BEST_MODEL_PATH = os.path.join(IDEA_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Target Columns
    # All 5 are predicted, but only specific ones are scored
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = 5

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Main Backbone (Dense Dilated TCN)
    GROWTH_RATE = 64
    LATENT_DIM = 64
    KERNEL_SIZE = 3
    DILATION_RATES = [1, 2, 4, 8, 16, 32]
    DROPOUT = 0.1

    # Global-Context Feedback Module
    FEEDBACK_DIM = 32
    FEEDBACK_GROWTH_RATE = 16

    # Interaction & Aggregation
    RNN_HIDDEN_DIM = 64  # Compact size

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 16  # Strictly set to 16
    LEARNING_RATE = 1e-3  # AdamW learning rate
    NUM_EPOCHS = 50  # Max epochs
    EARLY_STOPPING_PATIENCE = 10

    # Loss Configuration
    AUX_LOSS_WEIGHT = 0.5  # Weight for the first pass (no feedback) loss

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2
    SEED = 42
