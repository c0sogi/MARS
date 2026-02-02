import os
import torch


class Config:
    # =========================================================================
    # File Paths and Directories
    # =========================================================================
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Input directory (read-only)
    INPUT_DIR = "./input"
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working directory for artifacts (idea_13 specific)
    WORKING_DIR = "./working/idea_13"
    CACHE_DIR = WORKING_DIR  # Directory to store processed .npz/.pt files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Sequence dimensions
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # K-mer Tokenization
    K_MER_SIZE = 3
    # Vocab size: 4 bases (A,G,C,U) ^ 3 = 64.
    # Add 1 for padding/unknown if necessary.
    # We will map 3-mers to indices 1..64, 0 for padding.
    K_MER_VOCAB_SIZE = 64 + 1

    # Structure Tokenization
    # Characters: '(', ')', '.' -> 3 unique + 1 padding = 4
    STRUCTURE_VOCAB_SIZE = 4

    # Loop Type Tokenization
    # Characters: 'B', 'E', 'H', 'I', 'M', 'S', 'X' -> 7 unique + 1 padding = 8
    LOOP_VOCAB_SIZE = 8

    # Targets
    # We only train on the scored columns as per the idea description
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Columns to ignore during training (high noise)
    IGNORE_COLS = ["deg_pH10", "deg_50C"]

    # =========================================================================
    # Model Architecture (K-mer Enhanced Distance-Aware BiGRU)
    # =========================================================================
    HIDDEN_DIM = 384
    NUM_LAYERS = 5
    DROPOUT = 0.1
    USE_INPUT_INJECTION = True  # Flag to enable input injection logic

    # Embedding dimensions
    EMBED_DIM = 128  # Dimension for K-mer and other categorical embeddings

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 25
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0
    PATIENCE = 5  # Early stopping patience

    # Scheduler
    T_MAX = NUM_EPOCHS  # For CosineAnnealingLR

    # =========================================================================
    # Hardware & Reproducibility
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    SEED = 42


# Ensure working directory exists
os.makedirs(Config.WORKING_DIR, exist_ok=True)
