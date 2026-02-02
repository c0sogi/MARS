import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Metadata Files (Pre-split)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Files (if needed for reference)
    RAW_TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
    RAW_TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Outputs
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Hardware & Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # ==========================================
    # Model Hyperparameters (SS-DeGUT)
    # ==========================================
    # Architecture: Granular Unified Transformer
    D_MODEL = 256  # Embedding dimension
    N_LAYERS = 4  # Shallow depth for efficiency
    N_HEADS = 4  # Number of attention heads
    D_FF = 256 * 4  # Feed-forward dimension (1024)
    DROPOUT = 0.1  # Dropout rate
    ACTIVATION = "gelu"  # Activation function

    # Input specifics
    # 30 numerical features + engineered features
    # Sequence feature (f_27) length is 10
    MAX_SEQ_LEN = 64  # Safe upper bound for unified sequence (num + char tokens)
    VOCAB_SIZE = 40  # Estimate for character vocab (A-Z + specials)

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 2048  # Large batch size for A100
    EPOCHS = 50  # Max epochs
    LR = 1e-3  # Max learning rate (OneCycle)
    WEIGHT_DECAY = 1e-2  # AdamW weight decay

    # SS-DeGUT Specifics
    MASK_RATIO = 0.15  # Percentage of tokens to mask for reconstruction
    RECON_LOSS_WEIGHT = 1.0  # Lambda for reconstruction loss
    LABEL_SMOOTHING = 0.01  # Regularization for classification loss

    # Optimization
    PATIENCE = 5  # Early stopping patience
    PCT_START = 0.3  # OneCycleLR warmup percentage

    # ==========================================
    # Debugging
    # ==========================================
    DEBUG = False  # Set to True to run on a subset
    DEBUG_SAMPLES = 10000  # Number of samples to use in debug mode

    @classmethod
    def setup(cls):
        """
        Creates necessary working directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories created: {cls.WORKING_DIR}, {cls.SUBMISSION_DIR}")


def set_seed(seed=42):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")
