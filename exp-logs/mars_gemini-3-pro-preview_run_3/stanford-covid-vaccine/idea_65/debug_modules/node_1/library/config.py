import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration for High-Capacity Wide-Gated Decoupled BiGRU (HC-WG-BiGRU).
    Centralizes all file paths, model hyperparameters, and training settings.
    """

    # ==============================
    # File Paths & Directories
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_65"

    # Ensure working directory exists immediately
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Paths (Parquet files for efficient loading)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")

    # Raw Data Paths
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==============================
    # Data Configuration
    # ==============================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Input Features:
    # 4 (Nucleotide: A, G, C, U)
    # + 3 (Structure: ., (, ))
    # + 7 (Loop Type: S, M, I, B, H, E, X)
    INPUT_DIM = 14

    # Target Columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    NUM_TARGETS = 5

    # Indices of targets used for scoring:
    # 0: reactivity
    # 1: deg_Mg_pH10
    # 3: deg_Mg_50C
    SCORED_INDICES = [0, 1, 3]

    # ==============================
    # Model Hyperparameters (HC-WG-BiGRU)
    # ==============================
    # Backbone Specifications
    HIDDEN_DIM = 384  # Hidden size per direction (Total bidirectional size = 768)
    NUM_LAYERS = 4  # Deep 4-layer backbone
    BIDIRECTIONAL = True

    # Convolutional Stem
    CONV_FILTERS = 256
    CONV_KERNEL = 3

    # Wide Stabilized MLP Gate
    # The context input to the gate is (Hidden * 2 * 2) = 1536.
    # We project this to a WIDE dimension (768) rather than a bottleneck.
    GATE_HIDDEN_DIM = 768

    # Regularization
    DROPOUT = 0.1

    # ==============================
    # Training Hyperparameters
    # ==============================
    SEED = 42
    BATCH_SIZE = 16  # Adjusted for A100 memory with deep/wide model
    EPOCHS = 20  # Sufficient for convergence with early stopping
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0  # Gradient clipping for stability

    # Scheduler (Cosine Annealing)
    T_MAX = 20  # Matches EPOCHS
    ETA_MIN = 1e-5

    # Debugging / Development Flags
    DEBUG = False  # Set to True to train on a small subset
    SUBSET_SIZE = 100  # Number of samples if DEBUG is True

    # Compute
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    @staticmethod
    def set_seed(seed=42):
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
