import os
import torch


class Config:
    """
    Central configuration for the RNA Degradation Prediction task.
    Implements the 'Stabilized High-Capacity Wide-Stream BiLSTM' strategy.
    """

    # --------------------------------------------------------------------------
    # System & Reproducibility
    # --------------------------------------------------------------------------
    PROJECT_NAME = "idea_61"
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------
    # File Paths & Directories
    # --------------------------------------------------------------------------
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = f"./working/{PROJECT_NAME}"
    SUBMISSION_DIR = "./submission"

    # Create necessary writable directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache Files (for deterministic data processing)
    CACHE_TRAIN_PATH = os.path.join(WORKING_DIR, "train_data.pt")
    CACHE_VAL_PATH = os.path.join(WORKING_DIR, "val_data.pt")
    CACHE_TEST_PATH = os.path.join(WORKING_DIR, "test_data.pt")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    SEQ_LEN = 107
    SCORED_LEN = 68

    # Target Columns: We train ONLY on the 3 columns that are actually scored.
    # deg_pH10 and deg_50C are excluded to reduce noise.
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Debugging / Development
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100  # Number of samples to use when DEBUG is True

    # --------------------------------------------------------------------------
    # Model Architecture
    # Strategy: Stabilized Wide-Stream BiLSTM
    # --------------------------------------------------------------------------
    # Input Embeddings (Heterogeneous Feature Embedding)
    EMB_SEQ_DIM = 100  # Atomic Sequence (A, G, C, U) - Cite solution_lesson_node_00099
    EMB_LOOP_DIM = 64  # Predicted Loop Type
    EMB_PAIR_DIM = 64  # Signed Sinusoidal Pairing Distance

    # Fused Input Dimension: 100 + 64 + 64 = 228
    INPUT_DIM = EMB_SEQ_DIM + EMB_LOOP_DIM + EMB_PAIR_DIM

    # Backbone: Wide-Stream Residual BiLSTM
    HIDDEN_DIM = (
        384  # Optimized capacity for small data - Cite solution_lesson_node_00131
    )
    NUM_LAYERS = 6  # Shallow and Wide principle
    DROPOUT = (
        0.1  # Reduced dropout to preserve signal - Cite solution_lesson_node_00112
    )
    STEM_DROPOUT = 0.0  # No dropout in the initial projection stem
    BIDIRECTIONAL = True

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32  # Strictly 32 to maintain gradient quality
    EPOCHS = 20  # Fixed budget

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Low weight decay to preserve recurrent signals

    # Stabilization
    MAX_GRAD_NORM = 1.0  # Critical for 512-width BiLSTM convergence

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6
