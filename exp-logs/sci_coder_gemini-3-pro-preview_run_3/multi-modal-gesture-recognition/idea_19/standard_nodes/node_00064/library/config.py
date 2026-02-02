import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache Directory for Idea 19
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_19")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Submission
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Data Processing
    # ==========================================
    # Sliding Window Parameters
    WINDOW_SIZE = 64
    STRIDE_TRAIN = 32
    STRIDE_TEST = 32  # 50% overlap for inference (64/2)

    # Class Definitions
    # 20 Gestures + 1 Background
    NUM_CLASSES = 21
    BACKGROUND_CLASS_ID = 0
    # Original labels are 1-20. We map 0 to background.

    # Input Feature Dimensions
    # Audio: 13 MFCCs (standard) or similar.
    # Skeleton: 20 joints * 3 coords * 3 derivatives (pos, vel, acc) = 180
    # We will compute these dynamically in the dataset class, but defining here for model init.
    # Assuming 20 joints, 3D coords.
    NUM_JOINTS = 20
    # Features: (Pos + Vel + Acc) * 3 dims = 9 per joint. Total 180.
    # Plus Audio MFCC (e.g. 12 or 13). Let's assume 13 for now, adjustable in dataset.
    SKELETON_INPUT_DIM = NUM_JOINTS * 3 * 3  # 180
    AUDIO_INPUT_DIM = 13
    TOTAL_INPUT_DIM = SKELETON_INPUT_DIM + AUDIO_INPUT_DIM  # 193

    # ==========================================
    # Model Architecture (ASK-RN)
    # ==========================================
    # Stage 1: Bi-GRU Encoder
    HIDDEN_DIM = 128
    GRU_LAYERS = 2
    DROPOUT = 0.3

    # Stage 2 & 3: TCN Refinement
    TCN_NUM_CHANNELS = 64
    TCN_KERNEL_SIZE = 3
    # Dilation schedule as per idea: [1, 2, 4, 8, 16]
    TCN_DILATIONS = [1, 2, 4, 8, 16]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    LEARNING_RATE = 1e-3  # Adam default
    BATCH_SIZE = 32
    NUM_EPOCHS = 50

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # Loss Weights
    # Background class weight (0.2) vs others (1.0)
    CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
    CLASS_WEIGHTS[BACKGROUND_CLASS_ID] = 0.2

    # Multi-Task Loss Components
    LAMBDA_BOUNDARY = 0.5  # Weight for auxiliary boundary loss
    LAMBDA_SMOOTH = 0.15  # Weight for smoothing loss
    SMOOTH_LOSS_THRESHOLD = 1.0  # Truncated MSE threshold

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For data loading


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Automatically set seed when module is imported to guarantee consistency
seed_everything(Config.SEED)
