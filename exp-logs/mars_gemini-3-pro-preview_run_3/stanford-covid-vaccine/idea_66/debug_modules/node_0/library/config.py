import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # Reproducibility & Environment
    # ==========================================
    SEED = 42
    NUM_WORKERS = 2  # Adjust based on vCPUs (12 available)

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_66"

    # Input Data (Parquet Metadata)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Data Dimensions & Features
    # ==========================================
    SEQ_LEN = 107
    SEQ_SCORED = 68

    # Input Channels: 4 (Nucleotide) + 3 (Structure) + 7 (Loop Type)
    # Nucleotides: A, G, C, U
    # Structure: (, ), .
    # Loop Type: S, M, I, B, H, E, X
    INPUT_CHANNELS = 14

    # Target Columns
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Only these are used for the competition metric validation
    SCORING_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # ==========================================
    # Model Hyperparameters
    # Strategy: High-Capacity Full-Rank Decoupled BiGRU
    # ==========================================
    HIDDEN_DIM = 768  # High Capacity Backbone (384 per direction)
    NUM_LAYERS = 4  # 4-Layer Backbone
    GATE_HIDDEN_DIM = 768  # Full-Rank Gate (No bottleneck)

    CONV_FILTERS = 256  # Convolutional Stem filters
    KERNEL_SIZE = 3  # Convolutional Stem kernel size
    DROPOUT = 0.1  # Regularization

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_EPOCHS = 35
    EARLY_STOPPING_PATIENCE = 7
    GRAD_CLIP_NORM = 1.0  # Mandatory for stability

    # Debugging / Development
    DEBUG = False  # Set to True to use a small subset
    DEBUG_SUBSET_SIZE = 50

    @staticmethod
    def set_seed(seed=42):
        """Sets fixed random seeds for reproducibility."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def setup_directories(cls):
        """Ensures working and cache directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)


# Automatically setup environment on import
Config.setup_directories()
Config.set_seed(Config.SEED)
