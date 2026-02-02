import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the Modality-Aware 2.5D EfficientNet pipeline.
    """

    # --------------------------------------------------------------------------
    # 1. Directories & Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching and model checkpoints
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache File (for deterministic data processing if needed)
    CACHE_PATH = os.path.join(WORKING_DIR, "processed_data_cache.parquet")

    # --------------------------------------------------------------------------
    # 2. Data Processing Parameters
    # --------------------------------------------------------------------------
    # Image dimensions for EfficientNet-B0
    IMG_SIZE = 224

    # Modalities available in the dataset
    MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]

    # Slice Selection Strategy
    # We select 3 slices per modality: [Anchor - Stride, Anchor, Anchor + Stride]
    NUM_SLICES_PER_MODALITY = 3
    SLICE_STRIDE = 5

    # Total input channels = 4 modalities * 3 slices = 12
    TOTAL_CHANNELS = len(MODALITIES) * NUM_SLICES_PER_MODALITY

    # --------------------------------------------------------------------------
    # 3. Model Architecture Parameters
    # --------------------------------------------------------------------------
    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True
    NUM_CLASSES = 1
    DROPOUT_RATE = 0.2

    # Grouped Convolution for the first layer to separate modalities
    # Groups = 4 (one group per modality)
    FIRST_CONV_GROUPS = 4

    # --------------------------------------------------------------------------
    # 4. Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4  # Reduced from 1e-2 to avoid extinguishing weak signals (Cite Lesson 00003)

    # Early Stopping
    PATIENCE = 5

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging
    DEBUG = False  # Set to True to run on a small subset for testing


def seed_everything(seed=42):
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
