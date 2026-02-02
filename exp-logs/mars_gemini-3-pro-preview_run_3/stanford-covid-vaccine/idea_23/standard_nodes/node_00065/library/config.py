import os
import torch
import numpy as np
import random


class Config:
    # --------------------------------------------------------------------------
    # System & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Moderate number of workers for data loading
    NUM_WORKERS = 2

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_23"

    # Ensure the specific working directory exists immediately upon config load
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata paths (Parquet files generated in previous steps)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")

    # Submission and Reference paths
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Model Checkpoint
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache paths for deterministic data loading
    # Using .npz for efficient storage of processed tensors
    CACHE_TRAIN = os.path.join(WORKING_DIR, "train_data_cache.npz")
    CACHE_VAL = os.path.join(WORKING_DIR, "val_data_cache.npz")
    CACHE_TEST = os.path.join(WORKING_DIR, "test_data_cache.npz")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    SEQ_LEN = 107
    PRED_LEN = 68

    # Input Feature Dimensions
    # 4 (Nucleotide: A, G, C, U)
    # + 3 (Structure: (, ), .)
    # + 7 (Loop Type: S, M, I, B, H, E, X)
    INPUT_DIM = 14

    # Target Columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Columns specifically used for the competition scoring metric
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = 5

    # --------------------------------------------------------------------------
    # Model Hyperparameters (SP-MHA-BiGRU)
    # --------------------------------------------------------------------------
    # Hidden dimension constrained to 384 to prevent optimization failures
    HIDDEN_DIM = 384
    NUM_LAYERS = 3  # 3 Blocks of BiGRU + Attention
    NUM_HEADS = 4  # Multi-Head Attention heads (384 / 4 = 96 dim per head)
    DROPOUT = 0.1

    # Convolutional Stem parameters
    CNN_FILTERS = 256
    CNN_KERNEL = 3

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 25  # Sufficient for convergence with Cosine Annealing
    PATIENCE = 5  # Early stopping patience

    # Optimization constraints
    GRAD_CLIP = 1.0  # Mandatory clipping to stabilize hybrid architecture
    MIN_LR = 1e-5  # Minimum LR for Cosine Annealing scheduler

    # --------------------------------------------------------------------------
    # Debugging
    # --------------------------------------------------------------------------
    DEBUG = False
    DEBUG_SIZE = 100  # Subset size when DEBUG is True


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
