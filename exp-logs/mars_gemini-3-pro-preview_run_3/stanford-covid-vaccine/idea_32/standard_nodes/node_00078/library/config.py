import os
import torch


class Config:
    """
    Configuration for the SDIN-CG-BiGRU (Stabilized Deep Internally-Normalized
    Channel-Gated BiGRU) strategy.

    This class centralizes all hyperparameters, file paths, and constant definitions
    to ensure consistency across the data pipeline, model definition, and training loop.
    """

    # =========================================================================
    # Project & Experiment Settings
    # =========================================================================
    PROJECT_NAME = "RNA_Degradation_Prediction"
    IDEA_NAME = "idea_32"
    SEED = 42
    VERBOSE = True

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Source Data Paths (Parquet Metadata)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directory (Read/Write)
    # All artifacts for this specific idea/experiment are stored here
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Cache Directory for processed tensors
    # Used to store adjacency matrices and one-hot encoded features
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # =========================================================================
    # Data Specifications
    # =========================================================================
    # Sequence Dimensions
    SEQ_LEN = 107
    SEQ_SCORED = 68  # Only the first 68 positions are scored

    # Input Feature Channels (Total: 14)
    # 4 Nucleotides (A, G, C, U)
    # 3 Structure states ((, ), .)
    # 7 Loop types (S, M, I, B, H, E, X)
    INPUT_DIM = 14

    # Target Definitions
    # All 5 targets are predicted
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Only these 3 are used for the competition metric (MCRMSE)
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    OUTPUT_DIM = len(TARGET_COLS)

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Architecture: Stabilized Deep Internally-Normalized Channel-Gated BiGRU
    HIDDEN_DIM = 384  # High capacity width
    NUM_LAYERS = 4  # Deep backbone
    KERNEL_SIZE = 3  # For the convolutional stem
    DROPOUT = 0.1  # Regularization

    # =========================================================================
    # Training Configuration
    # =========================================================================
    BATCH_SIZE = 32  # Adjusted for A100 GPU with 384 hidden dim
    NUM_WORKERS = 4  # Data loading workers

    # Optimization
    LEARNING_RATE = 1e-3  # Standard for AdamW
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0  # Critical for stabilizing deep RNNs

    # Scheduler (Cosine Annealing)
    T_MAX = 50  # Matches EPOCHS
    ETA_MIN = 1e-6

    # Loop Controls
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print(f"Configuration for {cls.IDEA_NAME}:")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Model: {cls.NUM_LAYERS}x BiGRU layers, {cls.HIDDEN_DIM} hidden dim")
        print(f"  Input Dim: {cls.INPUT_DIM}, Output Dim: {cls.OUTPUT_DIM}")
        print(
            f"  Training: BS={cls.BATCH_SIZE}, LR={cls.LEARNING_RATE}, Epochs={cls.EPOCHS}"
        )
        print(f"  Paths: WorkDir={cls.WORKING_DIR}")
