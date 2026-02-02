import os
import torch


class Config:
    """
    Configuration class for the RNA Degradation Prediction task.
    Implements the 'High-Capacity GLU-Refined Decoupled BiGRU' strategy settings.
    """

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_69"
    SUBMISSION_DIR = "./submission"

    # Create working and submission directories if they don't exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Raw Input Files (JSON/CSV)
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Parquet - Pre-split and Stratified)
    TRAIN_PARQUET = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PARQUET = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PARQUET = os.path.join(METADATA_DIR, "test.parquet")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Directory for processed tensors
    CACHE_DIR = WORKING_DIR

    # =========================================================================
    # Data Configuration
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Input Features: 4 (A,G,C,U) + 3 (Structure: ., (, )) + 7 (LoopType)
    INPUT_DIM = 14

    # Targets
    NUM_TARGETS = 5
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Validation Metric Columns (Only these are scored in the competition metric)
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # =========================================================================
    # Model Architecture (High-Capacity GLU-Refined Decoupled BiGRU)
    # =========================================================================
    # Convolutional Stem
    STEM_KERNEL_SIZE = 3
    STEM_FILTERS = 256

    # Backbone: 4-Layer Bidirectional GRU
    # Hidden dim 384 per direction => 768 total output dimension
    HIDDEN_DIM = 384
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # Structural Interaction Module
    USE_GLU = True
    # Wide Gate Dimension: Projects input to a high-dimensional space before gating
    GATE_WIDE_DIM = 768

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 64  # Optimized for A100 40GB
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Gradient Clipping (Mandatory for stability)
    CLIP_GRAD_NORM = 1.0

    # Scheduler (Cosine Annealing)
    T_MAX = 50
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 10

    # =========================================================================
    # Runtime & Debugging
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging Flags
    # Set DEBUG = True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100

    @classmethod
    def get_config_dict(cls):
        """Returns the configuration as a dictionary."""
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
