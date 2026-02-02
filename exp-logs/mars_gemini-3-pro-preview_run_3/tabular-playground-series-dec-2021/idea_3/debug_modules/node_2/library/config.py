import os
import torch
import random
import numpy as np


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Config:
    # --------------------------------------------------------------------------
    # General Setup
    # --------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of workers for data loading

    # --------------------------------------------------------------------------
    # Directories & Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Input Data Sources (Metadata Parquet Files)
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Caching Paths (For Deterministic Data Processing)
    # These paths store the processed numpy arrays to speed up subsequent runs
    CACHE_TRAIN_X = os.path.join(WORKING_DIR, "train_X.npy")
    CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "train_y.npy")
    CACHE_VAL_X = os.path.join(WORKING_DIR, "val_X.npy")
    CACHE_VAL_Y = os.path.join(WORKING_DIR, "val_y.npy")
    CACHE_TEST_X = os.path.join(WORKING_DIR, "test_X.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "dcn_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Dataset Parameters
    # --------------------------------------------------------------------------
    TARGET_COL = "Cover_Type"
    ID_COL = "Id"

    # Feature Engineering: Prefixes for binary columns to be reconstructed
    SOIL_PREFIX = "Soil_Type"
    WILDERNESS_PREFIX = "Wilderness_Area"

    # --------------------------------------------------------------------------
    # Model Architecture (DCN-V2)
    # --------------------------------------------------------------------------
    # Entity Embeddings Configuration
    # Soil_Type: 40 binary columns -> 1 categorical index (cardinality ~40)
    # Wilderness_Area: 4 binary columns -> 1 categorical index (cardinality 4)
    EMBEDDING_CONFIG = {
        "soil": {"num_embeddings": 41, "embedding_dim": 16},  # +1 for safety/padding
        "wilderness": {
            "num_embeddings": 5,
            "embedding_dim": 4,
        },  # +1 for safety/padding
    }

    # Cross Network
    NUM_CROSS_LAYERS = 3

    # Deep Network (ResNet Backbone)
    HIDDEN_UNITS = [256, 256]  # List of hidden layer sizes
    DROPOUT_RATE = 0.15

    # Output
    NUM_CLASSES = 7  # Target classes range 1-7

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 2048
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    EPOCHS = 30
    PATIENCE = 5  # Early stopping patience

    # Debugging / Development
    # Set to a specific integer (e.g., 10000) to subsample the dataset for fast debugging
    # Set to None to use the full dataset
    DEBUG_SAMPLE_SIZE = None

    @classmethod
    def setup(cls):
        """
        Initializes the environment by creating necessary directories and setting seeds.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        set_seed(cls.SEED)
