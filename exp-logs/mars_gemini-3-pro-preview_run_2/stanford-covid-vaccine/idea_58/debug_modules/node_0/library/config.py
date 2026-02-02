import os
import torch


class Config:
    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_58"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Cache Paths (Explicit Cache Invalidation keys)
    CACHE_TRAIN = os.path.join(WORKING_DIR, "train_data_gc_ssn_v1.npz")
    CACHE_VAL = os.path.join(WORKING_DIR, "val_data_gc_ssn_v1.npz")
    CACHE_TEST = os.path.join(WORKING_DIR, "test_data_gc_ssn_v1.npz")

    # Submission Path
    SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # =========================================================================
    # Data Specifications
    # =========================================================================
    SEQ_LEN = 107
    SCORED_LEN = 68

    # Columns
    # Scored targets for the competition metric
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # Unscored targets (provided in training but not scored, used for feedback masking)
    UNSCORED_COLS = ["deg_pH10", "deg_50C"]
    # All target columns in order
    TARGET_COLS = SCORED_COLS + UNSCORED_COLS

    # Vocabulary mappings
    BASES = ["A", "G", "C", "U"]
    STRUCTS = [".", "(", ")"]
    LOOPS = ["S", "M", "I", "B", "H", "E", "X"]

    # =========================================================================
    # Model Hyperparameters (GC-SSN)
    # =========================================================================
    # Input Dim: 4 (Seq) + 3 (Struct) + 7 (Loop) + 4 (Partner Identity) = 18
    INPUT_DIM = 18

    # Backbone (Dense Dilated TCN)
    HIDDEN_DIM = 64
    GROWTH_RATE = 64
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4, 8, 16, 32]
    DROPOUT = 0.1

    # Global-Context Pure-Feedback Module
    FEEDBACK_INPUT_DIM = 5  # 5 target channels
    FEEDBACK_DIM = 32
    FEEDBACK_GROWTH_RATE = 16

    # Aggregation
    RNN_HIDDEN = 64

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 16  # Strictly set to 16 as per requirements
    LR = 1e-3
    EPOCHS = 50
    PATIENCE = 10  # For Early Stopping

    # Reproducibility
    SEED = 42

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    # =========================================================================
    # Debugging & Development
    # =========================================================================
    # Set to True to run on a small subset of data
    DEBUG = False
    MAX_DEBUG_SAMPLES = 100
