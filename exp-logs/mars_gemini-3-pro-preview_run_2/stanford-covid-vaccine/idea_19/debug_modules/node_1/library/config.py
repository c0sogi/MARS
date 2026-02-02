import os
import torch
import numpy as np
import random


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Config:
    # ==============================
    # Paths & Directories
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_19"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission Path
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==============================
    # Data Configuration
    # ==============================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Target Columns
    # Scored targets for the objective function (Masked Optimization)
    # We only compute loss on these 3 columns as per the metric definition
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # All targets provided in training data (for parsing)
    ALL_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Cache Versioning
    # Explicit cache invalidation for "Asymmetric Dense-Context Hybrid Network"
    # Ensures Partner Identity features are generated fresh and not loaded from stale caches.
    CACHE_VERSION = "asymmetric_dense_v1"

    # ==============================
    # Model Hyperparameters
    # ==============================
    # Backbone: Dense Dilated TCN
    HIDDEN_DIM = 64  # Growth Rate / Channel Width
    DROPOUT = 0.1
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4, 8, 16, 32]  # Exponential dilation for global receptive field

    # Asymmetric Latent Fusion Module
    LOCAL_PROJ_DIM = 128  # Stream 1: Local Fidelity (High dim)
    STRUCT_PROJ_DIM = (
        64  # Stream 2: Structural Context (Low dim, gathered from partner)
    )

    # Global Aggregation (BiGRU)
    # Note: RNN Hidden size is strictly input_dim // 2 to match output dim to input.
    # Input to RNN is LOCAL_PROJ_DIM + STRUCT_PROJ_DIM = 192.
    # Therefore, RNN hidden size will be 96, resulting in 192 output features.

    # ==============================
    # Training Hyperparameters
    # ==============================
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 25  # Sufficient for convergence with this dataset size
    NUM_WORKERS = 4
    SEED = 42

    # Optimization
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0

    # Scheduler
    PATIENCE = 4
    FACTOR = 0.5
