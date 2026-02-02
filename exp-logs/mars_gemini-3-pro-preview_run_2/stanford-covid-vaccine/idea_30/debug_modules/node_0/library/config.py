import os
import torch
import numpy as np
import random


class Config:
    # ==============================
    # File Paths & Directories
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_30"

    # Cache Filenames (Versioned to ensure data consistency with Idea 30)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_sr_dcn_v1.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_sr_dcn_v1.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_sr_dcn_v1.npz")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # ==============================
    # Data Specifications
    # ==============================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Feature Dimensions
    # Sequence (4) + Structure (3) + Loop (7) + PartnerID (4) + Recycling (5)
    INPUT_CHANNELS = 4 + 3 + 7 + 4 + 5

    # ==============================
    # Model Architecture (SR-DCN)
    # ==============================
    HIDDEN_DIM = 64  # Base channel width
    GROWTH_RATE = 64  # Dense connection growth rate
    LATENT_DIM = 64  # Dimension for structural interaction
    DROPOUT = 0.1
    KERNEL_SIZE = 3

    # Dilated TCN Backbone
    DILATION_RATES = [1, 2, 4, 8, 16, 32]

    # ==============================
    # Training Hyperparameters
    # ==============================
    SEED = 42
    BATCH_SIZE = 16
    EPOCHS = 25  # Sufficient for convergence with early stopping
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Loss Weights
    AUX_LOSS_WEIGHT = 0.5  # Weight for the Pass 1 (Cold Start) loss

    # Early Stopping
    PATIENCE = 5

    # Debugging/Development
    DEBUG = False  # Set to True to use a small subset of data
    NUM_WORKERS = 2

    @staticmethod
    def set_seed(seed=42):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Ensure working directory exists
os.makedirs(Config.WORKING_DIR, exist_ok=True)
os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
