import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Corrected Dual-Context Anchor Network (DC-AN).
    Defines paths, hyperparameters, and utility functions.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    @staticmethod
    def set_seed(seed=42):
        """Sets the seed for reproducibility across random, numpy, and torch."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Pre-computed Features (Parquet format)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Model Checkpoint & Submission
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Backbone for text embedding
    BACKBONE_NAME = "sentence-transformers/all-mpnet-base-v2"

    # Architecture dimensions
    HIDDEN_DIM = 512
    NHEAD = 8
    NUM_ENCODER_LAYERS = 2  # Layers for the Code and Markdown context transformers
    DROPOUT = 0.1

    # Max sequence length for the backbone tokenizer
    MAX_LENGTH = 128

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 5
    WEIGHT_DECAY = 0.01

    # Early Stopping
    PATIENCE = 3

    # Optimization
    WARMUP_RATIO = 0.0  # Disabled as per instructions

    # ==========================================
    # Compute Settings
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers

    # ==========================================
    # Debugging / Development
    # ==========================================
    # Set to an integer (e.g., 1000) to limit the dataset size for faster debugging.
    # Set to None to use the full dataset.
    DEBUG_SAMPLE_SIZE = None
