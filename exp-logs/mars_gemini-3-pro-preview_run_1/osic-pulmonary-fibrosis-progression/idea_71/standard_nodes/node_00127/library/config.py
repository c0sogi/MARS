import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # 1. File Paths & Directories
    # ==========================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific Metadata Files (Generated previously)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # DICOM Directories
    TRAIN_DICOM_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DICOM_DIR = os.path.join(INPUT_DIR, "test")

    # Caching & Artifacts
    # Specific cache directory for this idea
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_71")
    MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission", "submission.csv")

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==========================================
    # 2. Data Preprocessing & Augmentation
    # ==========================================
    # Image Configuration
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0
    SLAB_COUNT = 3  # Tri-Slab design
    OVERLAP = 0.15  # 15% overlap between slabs
    IN_CHANNELS = 3  # RGB (MIPs mapped to channels)

    # Normalization Constants (Derived from EDA)
    # Used to standardize tabular inputs
    TABULAR_MEAN = {"Age": 67.58, "Percent": 76.91}

    TABULAR_STD = {"Age": 6.63, "Percent": 19.20}

    # Categorical Encodings
    SEX_MAP = {"Male": 0, "Female": 1}
    SMOKING_MAP = {"Ex-smoker": 0, "Never smoked": 1, "Currently smokes": 2}

    # ==========================================
    # 3. Model Architecture Hyperparameters
    # ==========================================
    # Backbone
    BACKBONE_NAME = "efficientnet_b0"
    PRETRAINED = True
    BACKBONE_DIM = 1280  # Feature dimension of B0 before classifier

    # Shared Latent Topology
    LATENT_DIM = 128  # Dimension for T_lat and H_update

    # Attention Mechanism
    NUM_HEADS = 4
    DROPOUT = 0.1

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 16  # Adjusted for 224x224 dual-view on available VRAM
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler
    T_MAX = 50  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 8  # Strict patience as per design

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 5. Inference & Metric
    # ==========================================
    # Metric Constants
    MAX_ERROR = 1000  # Error clipping threshold (ml)
    MIN_CONFIDENCE = 70  # Confidence clipping threshold (ml)

    # Inference Batch Size (can be larger as no gradients)
    INFERENCE_BATCH_SIZE = 32


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
