import os
import torch


class Config:
    """
    Configuration class for RNA Degradation Prediction using Graph Neural Networks.
    Centralizes all file paths, model hyperparameters, and training settings.
    """

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Input Metadata Paths (Parquet files)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Sample Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_PATH = "submission.csv"
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache Paths (for processed Graph Data objects)
    # Using .pt extension for PyTorch Geometric data lists
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_graphs.pt")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_graphs.pt")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_graphs.pt")

    # -------------------------------------------------------------------------
    # Data Specifications
    # -------------------------------------------------------------------------
    SEQ_LEN = 107
    SCORED_LEN = 68

    # Target Columns used for scoring and training
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    NUM_TARGETS = len(TARGET_COLS)

    # Indices of targets that are actually scored in the competition
    # reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    SCORED_INDICES = [0, 1, 3]

    # Vocabulary Mappings
    # Sequence: A, G, U, C
    VOCAB_MAP_SEQ = {"A": 0, "G": 1, "U": 2, "C": 3}
    VOCAB_SIZE_SEQ = 4

    # Structure: (, ), .
    VOCAB_MAP_STRUCT = {"(": 0, ")": 1, ".": 2}
    VOCAB_SIZE_STRUCT = 3

    # Predicted Loop Type: S, M, I, B, H, E, X
    VOCAB_MAP_LOOP = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
    VOCAB_SIZE_LOOP = 7

    # -------------------------------------------------------------------------
    # Model Hyperparameters (R-GNN)
    # -------------------------------------------------------------------------
    # Dimension of embeddings for sequence, structure, and loop type
    EMBED_DIM = 64

    # Hidden dimension for GNN layers
    HIDDEN_DIM = 256

    # Number of Graph Convolution/Attention layers
    NUM_LAYERS = 4

    # Dropout rate
    DROPOUT = 0.3

    # Number of edge types (Backbone + BasePair)
    # 0: Backbone (i -> i+1), 1: Backbone (i+1 -> i), 2: Base Pair
    NUM_EDGE_TYPES = 3

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # A100 has 40GB VRAM, allowing for larger batch sizes
    BATCH_SIZE = 64

    # Learning Rate
    LR = 1e-3

    # Weight Decay for regularization
    WEIGHT_DECAY = 1e-4

    # Learning Rate Scheduler settings (Cosine Annealing)
    T_MAX = 50  # Should match EPOCHS usually
    ETA_MIN = 1e-6

    # Training duration
    EPOCHS = 50

    # Early Stopping
    PATIENCE = 10  # Stop if validation loss doesn't improve for 10 epochs

    # -------------------------------------------------------------------------
    # Hardware & Computation
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers
