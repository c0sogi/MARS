import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the AASL-Net pipeline (Idea 72).
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Input Subdirectories
    TRAIN_DIR = os.path.join(INPUT_ROOT, "train")
    TEST_DIR = os.path.join(INPUT_ROOT, "test")

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sample_submission.csv")

    # Output/Working Directories
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_72")
    SUBMISSION_DIR = "./submission"

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model_idea_72.pth")

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0
    N_SLABS = 3  # Tri-Slab configuration
    OVERLAP = 0.15  # 15% overlap between slabs
    USE_CACHE = True  # Enable caching of processed tensors

    # ==========================================
    # Model Architecture Constants
    # ==========================================
    BACKBONE_NAME = "efficientnet_b0"
    BACKBONE_DIM = 1280  # Output dim of EfficientNet-B0 GAP
    LATENT_DIM = 128  # Shared latent dim for tabular and fusion
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32  # Conservative batch size for A100 with dual backbones
    EPOCHS = 50  # Max epochs, controlled by early stopping
    LEARNING_RATE = 1e-4  # AdamW learning rate
    WEIGHT_DECAY = 1e-2
    PATIENCE = 8  # Strict patience for early stopping
    NUM_WORKERS = 4  # 12 vCPUs available

    # ==========================================
    # Metric / Loss Constants
    # ==========================================
    MAX_ERROR = 1000.0  # Clipping threshold for metric
    MIN_CONFIDENCE = 70.0  # Clipping threshold for sigma

    # ==========================================
    # Debugging / Development
    # ==========================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 20  # Number of patients to use in debug mode

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.WORKING_DIR, exist_ok=True)


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


# Initialize directories and seeds upon import
Config.setup()
seed_everything(Config.SEED)
