import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific strategy (Idea 19)
    WORKING_DIR = "./working/idea_19"
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths (Parquet)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (for precomputed features/graphs)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_cache.npy")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_cache.npy")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_cache.npy")

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Specifications
    # ==========================================
    SEQ_LEN = 107
    SEQ_SCORED = 68

    # Input Channels:
    # 4 (Nucleotide: A, G, U, C)
    # + 3 (Structure: ., (, ))
    # + 7 (Loop Type: S, M, I, B, H, E, X)
    NUM_NODE_FEATURES = 14

    # Target Columns for Scoring
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    NUM_TARGETS = 5

    # ==========================================
    # Model Architecture: DASR-BiGRU
    # ==========================================
    # Convolutional Stem
    CONV_FILTERS = 256
    CONV_KERNEL_SIZE = 3

    # Recurrent Backbone
    HIDDEN_DIM = 384
    NUM_LAYERS = 3  # Number of BiGRU + Interaction Blocks

    # Distance Awareness Mechanism
    DISTANCE_EMBEDDING_DIM = 32
    MAX_DISTANCE = 128  # Cap for relative distance embedding (covers 107 length)

    # Regularization
    DROPOUT = 0.1
    EDGE_DROPOUT = 0.15  # Probability to drop structural connections during training

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 7

    # Gradient Clipping to stabilize Deep BiGRU
    GRAD_CLIP_NORM = 1.0

    # Scheduler Settings (Cosine Annealing)
    T_MAX = EPOCHS

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLES = 100  # Number of samples to use if DEBUG is True

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
