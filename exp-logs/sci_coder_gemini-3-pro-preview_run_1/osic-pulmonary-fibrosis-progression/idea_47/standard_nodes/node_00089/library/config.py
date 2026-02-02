import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_47"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata file paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Model checkpoint path
    MODEL_PATH = os.path.join(WORKING_DIR, "slh_dan_best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Cache paths for processed images (if used by data loader)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ==========================================
    # 2. Data & Image Processing
    # ==========================================
    # Image dimensions
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0

    # Tri-Slab generation parameters
    SLAB_COUNT = 3
    SLAB_OVERLAP = 0.15  # 15% overlap

    # Normalization (ImageNet stats)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ==========================================
    # 3. Model Architecture (SLH-DAN)
    # ==========================================
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True

    # Dimensionality
    BACKBONE_OUT_DIM = 1280  # EfficientNet-B0 output channels (GAP)
    LATENT_DIM = 128  # Shared Latent Vector (T_lat)
    PROJECTED_DIM = 1280  # T_align (matches backbone)

    # Tabular features
    TABULAR_COLS = ["Age", "Sex", "SmokingStatus", "Percent"]
    NUM_TABULAR_FEATURES = 9  # Age, Sex(2), Smoking(3), Percent, plus embeddings if any
    # Note: Actual input dim depends on encoding (OneHot etc.) handled in dataset class

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 16
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler
    T_MAX = 30  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 8

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 5. Metric & Loss Constants
    # ==========================================
    # Modified Laplace Log Likelihood constants
    MAX_ERROR = 1000
    MIN_CONFIDENCE = 70


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
