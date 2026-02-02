import os
import torch


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"

    # Ensure the working directory for this idea exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    SEQ_SCORED = 68

    # Target Columns
    # ALL_TARGET_COLS: Used for parsing the dataset (all available ground truth)
    ALL_TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # SCORED_TARGET_COLS: Used for Loss Calculation and MCRMSE Metric
    # As per task description, only these 3 are scored.
    SCORED_TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Vocabulary Definitions for One-Hot Encoding
    BASES = ["A", "G", "C", "U"]
    STRUCTURES = [".", "(", ")"]
    LOOP_TYPES = ["S", "M", "I", "B", "H", "E", "X"]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # For Early Stopping
    NUM_WORKERS = 2

    # =========================================================================
    # Model Architecture Hyperparameters
    # =========================================================================
    # Input dimension will be dynamic based on feature flags, but hidden dim is fixed
    HIDDEN_DIM = 128
    KERNEL_SIZE = 3
    DROPOUT = 0.2

    # Dilated TCN settings
    # Dilation rates will be 2^0, 2^1, ... 2^(NUM_LAYERS-1)
    NUM_LAYERS = 6

    # Architecture Feature Flags
    USE_PARTNER_FEATURES = True
    USE_DISTANCE_FEATURES = True
    USE_POSITIONAL_ENCODING = True

    # =========================================================================
    # Compute
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
