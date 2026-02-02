import os
import torch


class Config:
    # =========================================================================
    # Directories and Paths
    # =========================================================================
    # Base working directory for this specific idea/run
    WORKING_DIR = "./working/idea_78"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Input directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Data File Paths
    # Using the parquet metadata files as requested
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    SEQ_LEN = 107
    SEQ_SCORED = 68

    # Column Names
    ID_COL = "id"
    SEQUENCE_COL = "sequence"
    STRUCTURE_COL = "structure"
    LOOP_TYPE_COL = "predicted_loop_type"

    # Target Definitions
    # All 5 targets are predicted, but only 3 are scored in the metric
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = 5

    # =========================================================================
    # Model Architecture Configuration
    # =========================================================================
    # Input Features:
    # 4 Nucleotides (A, G, C, U)
    # + 3 Structure types ((, ), .)
    # + 7 Loop types (S, M, I, B, H, E, X)
    INPUT_DIM = 14

    # Convolutional Stem
    STEM_KERNEL_SIZE = 3
    STEM_FILTERS = 256

    # Backbone: Deep Residual High-Capacity BiGRU
    # Hidden dim is per direction. Total bidirectional output = 384 * 2 = 768
    HIDDEN_DIM = 384
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # Architecture Flags (Strategy Specific)
    USE_VERTICAL_RESIDUALS = True  # Enables skip connections across RNN layers
    USE_GLU_INTERACTION = True  # Enables Stabilized GLU-Decoupled Interaction Module

    # =========================================================================
    # Training Configuration
    # =========================================================================
    SEED = 42

    # Optimization
    BATCH_SIZE = 32  # Adjusted for deep model on A100
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Stability
    MAX_GRAD_NORM = 1.0  # Crucial for deep RNN stability

    # Early Stopping
    PATIENCE = 10

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
