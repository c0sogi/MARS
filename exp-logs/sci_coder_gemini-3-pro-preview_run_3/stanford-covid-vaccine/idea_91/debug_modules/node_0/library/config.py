import os
import torch


class Config:
    """
    Configuration for High-Capacity Motif-Aware Synthesis Strategy.
    Defines hyperparameters, file paths, and architectural settings.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_91"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Paths (Parquet files)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Raw Input Paths
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (for processed tensors if needed)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_cache.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_cache.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_cache.npz")

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Input Feature Dimensions
    # One-hot encoding: 4 (A,G,C,U) + 3 (Structure: .,(,)) + 7 (Loop: S,M,I,B,H,E,X)
    INPUT_DIM = 14

    # Target Columns (Ground Truth)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Columns used for Validation Scoring
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # =========================================================================
    # Model Architecture: High-Capacity BiGRU with Dilated Stem
    # =========================================================================
    # Dilated Residual Motif-Encoding Stem
    STEM_FILTERS = 384
    STEM_KERNEL_SIZE = 3
    STEM_DILATIONS = [1, 2, 4]  # 3-Stage Dilated Residual Network

    # Recurrent Backbone
    RNN_HIDDEN_DIM = 384  # Per direction (Total = 768)
    RNN_LAYERS = 4
    BIDIRECTIONAL = True

    # Interaction Module (Stabilized GLU-Decoupled)
    INTERACTION_DIM = 768  # Matches bidirectional output

    # Regularization
    DROPOUT = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    BATCH_SIZE = 32  # Conservative for large model on A100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_EPOCHS = 50
    PATIENCE = 10  # Early stopping patience

    # Stability
    GRAD_CLIP = 1.0  # Mandatory for hybrid architecture stability

    # Scheduler (Cosine Annealing)
    T_MAX = 50
    ETA_MIN = 1e-6

    # DataLoader
    NUM_WORKERS = 4
    PIN_MEMORY = True
