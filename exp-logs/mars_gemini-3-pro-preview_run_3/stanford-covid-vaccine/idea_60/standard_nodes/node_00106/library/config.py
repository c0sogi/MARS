import os
import torch
import numpy as np
import random


def seed_everything(seed: int = 42):
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


class Config:
    """
    Central configuration for the HC-SDBR-BiGRU model pipeline.
    """

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for idea_60 as specified in the caching requirements
    WORKING_DIR = "./working/idea_60"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    SEQ_LEN = 107
    SEQ_SCORED = 68

    # Input features:
    # 4 (A, G, C, U) +
    # 3 (Structure: (, ), .) +
    # 7 (Loop Type: S, M, I, B, H, E, X)
    INPUT_DIM = 14

    # Target columns for Multi-Task Learning (Training on all 5)
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Columns used for Validation Metric (MCRMSE)
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    NUM_TARGETS = len(TARGET_COLS)

    # ==========================================
    # Model Architecture (HC-SDBR-BiGRU)
    # ==========================================
    # Convolutional Stem
    STEM_KERNEL_SIZE = 3
    STEM_FILTERS = 256

    # Backbone
    # Hidden dimension per direction. Total hidden size = 384 * 2 = 768
    HIDDEN_DIM = 384
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 2024

    # Training settings
    NUM_EPOCHS = 30
    BATCH_SIZE = 16  # Adjusted for stability with high-capacity model
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Optimization Stability
    MAX_GRAD_NORM = 1.0  # Mandatory for hybrid architecture stability
    PATIENCE = 7  # Early stopping patience

    # Scheduler
    T_MAX = NUM_EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # ==========================================
    # Hardware & Debugging
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debug flag to limit dataset size for quick testing
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100
