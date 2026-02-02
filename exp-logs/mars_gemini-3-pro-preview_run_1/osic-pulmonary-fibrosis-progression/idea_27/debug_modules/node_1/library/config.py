import os
import torch
import numpy as np
import random


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


class Config:
    # ==========================================
    # 1. System & Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # ==========================================
    # 2. File Paths & Directories
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Input Metadata
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (for Caching & Checkpoints)
    WORKING_DIR = "./working"
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_27")
    CACHE_DIR = IDEA_DIR  # Cache processed images/features here

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(IDEA_DIR, "best_model.pth")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(IDEA_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 3. Data Processing & Augmentation
    # ==========================================
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0
    NUM_SLABS = 3  # Tri-Slab configuration
    SLAB_OVERLAP = 0.15  # 15% overlap between slabs

    # Tabular Features used in the MLP embedding
    STATIC_FEATURES = ["Age", "Sex", "SmokingStatus", "Percent"]

    # Debugging / Quick Run Controls
    DEBUG = False
    MAX_TRAIN_SAMPLES = None  # Set to int (e.g., 100) to debug pipeline speed

    # ==========================================
    # 4. Model Architecture (VC-DAN)
    # ==========================================
    BACKBONE = "efficientnet_b0"
    VISUAL_DIM = 1280  # Native B0 output dimension (no bottleneck)
    TABULAR_DIM = 1280  # Up-projected dimension for tabular data
    N_HEADS = 4  # Number of attention heads
    DROPOUT = 0.1  # Dropout rate

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    EPOCHS = 50
    BATCH_SIZE = 32  # Fits in A100 40GB with B0
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # For AdamW
    PATIENCE = 8  # Early stopping patience

    # ==========================================
    # 6. Metric & Loss Constants
    # ==========================================
    MAX_ERROR = 1000  # Clipping threshold for metric calculation
    MIN_CONFIDENCE = 70  # Minimum clipped confidence


# Apply seeding immediately
seed_everything(Config.SEED)
