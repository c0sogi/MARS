import os
import torch
import random
import numpy as np


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    """
    Central configuration for the Corrected Dual-Context Anchor Network (DC-AN).
    """

    # ==========================================
    # 1. General Settings
    # ==========================================
    EXP_ID = "idea_11"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data
    SAMPLE_SIZE = 1000  # Number of notebooks to use when DEBUG is True

    # ==========================================
    # 2. Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", EXP_ID)
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Feature Cache Paths (Parquet format)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 3. Model Architecture
    # ==========================================
    # Backbone for feature extraction
    MODEL_BACKBONE = "sentence-transformers/all-mpnet-base-v2"

    # Tokenization settings
    MAX_LENGTH = 128  # Truncation length for MPNet

    # Projection and Context settings
    PROJECTION_DIM = 512  # Shared latent space dimension
    HIDDEN_DIM = 512  # Hidden dimension for Transformer layers
    NHEAD = 8  # Number of attention heads
    NUM_LAYERS = 2  # Number of Transformer layers (Encoder)
    DROPOUT = 0.1

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3  # No warmup, constant LR
    WEIGHT_DECAY = 0.01
    PATIENCE = 3  # Early stopping patience

    # ==========================================
    # 5. Hardware and Compute
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
