import os
import torch
import numpy as np
import random


class Config:
    """
    Global configuration for the Robust Modality-Structured High-Density (RMS-HD) Network.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Cache Directory for deterministic data processing (Idea 24)
    CACHE_DIR = "./working/idea_24/"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission.csv"

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 224

    # High-Density Volumetric Settings
    # 32 slices per modality * 4 modalities = 128 input channels
    NUM_SLICES_PER_MODALITY = 32
    IN_CHANNELS = 4 * NUM_SLICES_PER_MODALITY

    # Data Loading
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # ==========================================
    # Model Configuration
    # ==========================================
    BACKBONE_NAME = "efficientnet_b0"

    # Stabilized Global-Mixing Adapter (Stem) Settings
    STEM_OUT_CHANNELS = 64  # Compress 128 -> 64

    # Regularization
    DROP_PATH_RATE = 0.2
    PRETRAINED = True

    # ==========================================
    # Training Configuration
    # ==========================================
    SEED = 42
    BATCH_SIZE = 8  # Conservative for A100 with large 128-channel inputs
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.0  # Disabled based on analysis (Adam vs AdamW)

    # Early Stopping
    PATIENCE = 5

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 32


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
