import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the InChI prediction task.
    """

    # --- Reproducibility ---
    SEED = 42

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"

    # Ensure working directory exists for caching and checkpoints
    os.makedirs(WORKING_DIR, exist_ok=True)

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # --- Data Hyperparameters ---
    IMAGE_SIZE = 384  # Fixed resolution as per baseline idea
    MAX_LEN = 450  # Max sequence length (EDA max was 403 + buffer for SOS/EOS)
    VOCAB_PATH = os.path.join(WORKING_DIR, "vocab.json")  # Path to cache vocabulary

    # --- Model Hyperparameters ---
    ENCODER_NAME = "resnet18"  # Lightweight CNN backbone

    # Transformer Decoder settings
    DECODER_DIM = 256  # Hidden dimension size (d_model)
    NUM_HEADS = 4  # Number of attention heads (as per idea)
    NUM_LAYERS = 3  # Number of decoder layers (shallow for speed)
    FF_DIM = 1024  # Feed-forward dimension
    DROPOUT = 0.1  # Dropout rate

    # --- Training Hyperparameters ---
    BATCH_SIZE = 64  # Batch size for A100 GPU
    NUM_WORKERS = 4  # Number of data loading workers
    LEARNING_RATE = 4e-4  # Max learning rate for OneCycleLR
    WEIGHT_DECAY = 1e-6  # Weight decay for AdamW
    NUM_EPOCHS = 10  # Total training epochs
    PATIENCE = 3  # Early stopping patience
    GRAD_CLIP = 1.0  # Gradient clipping value

    # --- Device ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        """Sets reproducible seeds."""
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Deterministic operations can be slower, but ensure reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Initialize environment
Config.setup()
