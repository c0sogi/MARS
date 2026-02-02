import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Checkpoints and Cache
    # We use idea_2 to separate this run's artifacts
    WORKING_DIR = "./working/idea_2"

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    # Image Resolution: Increased to 256x256 for better fine-grained detail
    IMG_SIZE = 256

    # Number of classes in the dataset
    NUM_CLASSES = 64500

    # DataLoader settings
    # Batch size 64 fits ConvNeXt-Small @ 256x256 on A100 comfortably with AMP
    BATCH_SIZE = 64
    NUM_WORKERS = 12  # Using available vCPUs

    # ==========================================
    # Model Configuration
    # ==========================================
    # Backbone architecture
    MODEL_NAME = "convnext_small"

    # Embedding dimension for ConvNeXt-Small is 768
    # This is significantly smaller than ResNet50's 2048, reducing head size
    EMBEDDING_DIM = 768

    # Pretrained weights
    PRETRAINED = True

    # ==========================================
    # Training Configuration
    # ==========================================
    # Total training duration
    EPOCHS = 12

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.05

    # Regularization
    LABEL_SMOOTHING = 0.1

    # Stochastic Weight Averaging (SWA)
    USE_SWA = True
    SWA_START_EPOCH = 8
    SWA_LR = 5e-5

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    # Mixed Precision
    USE_AMP = True

    @classmethod
    def setup(cls):
        """Ensure necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


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
    torch.backends.cudnn.benchmark = False  # Set to False for absolute reproducibility
