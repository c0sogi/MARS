import os
import torch
import numpy as np
import random


class Config:
    # =========================================================================
    # Paths
    # =========================================================================
    # Root directory for input data (Read-Only)
    INPUT_DIR = "./input"

    # Root directory for generated metadata (Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")

    # Working directory for caching and outputs
    # We use 'idea_4' as the specific folder for this iteration of the solution
    WORKING_DIR = "./working/idea_4"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data & Preprocessing
    # =========================================================================
    # Image Dimensions
    # Using 512x512 to balance detection resolution with EfficientNet-B0 compute requirements
    IMAGE_SIZE = (512, 512)  # (Height, Width)

    # DICOM Loading
    # If True, inverts pixel values if PhotometricInterpretation is MONOCHROME1
    FIX_MONOCHROME1 = True

    # Caching
    # Set to True to enable saving/loading processed images/masks to/from disk
    USE_CACHE = True

    # =========================================================================
    # Class Definitions
    # =========================================================================
    # 14 Findings + 1 "No finding" class
    CLASS_MAP = {
        0: "Aortic enlargement",
        1: "Atelectasis",
        2: "Calcification",
        3: "Cardiomegaly",
        4: "Consolidation",
        5: "ILD",
        6: "Infiltration",
        7: "Lung Opacity",
        8: "Nodule/Mass",
        9: "Other lesion",
        10: "Pleural effusion",
        11: "Pleural thickening",
        12: "Pneumothorax",
        13: "Pulmonary fibrosis",
        14: "No finding",
    }

    # Inverse mapping
    CLASS_NAME_TO_ID = {v: k for k, v in CLASS_MAP.items()}

    NUM_CLASSES = 15  # 0-13 are findings, 14 is no finding

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True

    # CoordConv: Add 2 channels (x, y coordinates) to features before detection head
    USE_COORD_CONV = True

    # BiFPN settings
    BIFPN_CHANNELS = 64

    # Global Classification Head settings
    # Threshold to gate the output. If P(No Finding) > THRESHOLD, output "14 1 0 0 1 1"
    GLOBAL_CLS_THRESHOLD = 0.8

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    NUM_EPOCHS = 20
    BATCH_SIZE = 8  # Adjusted for A100 40GB with 512x512 images

    # Optimizer
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Scheduler (Cosine Annealing)
    T_MAX = NUM_EPOCHS
    ETA_MIN = 1e-6

    # Loss Weights
    # Weights for the multi-task loss
    LAMBDA_HEATMAP = 1.0
    LAMBDA_SIZE = 0.1
    LAMBDA_OFFSET = 0.1
    LAMBDA_GLOBAL = 1.0  # Weight for the global classification head

    # =========================================================================
    # Augmentation
    # =========================================================================
    # Geometric
    AUG_SHIFT_LIMIT = 0.0625
    AUG_SCALE_LIMIT = 0.1
    AUG_ROTATE_LIMIT = 15
    # Critical: ensures we don't crop out the object completely while keeping the label
    AUG_MIN_VISIBILITY = 0.3

    # Photometric
    AUG_BRIGHTNESS_LIMIT = 0.2
    AUG_CONTRAST_LIMIT = 0.2
    # Explicitly disabling CLAHE as per lessons learned
    USE_CLAHE = False

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers


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

    # Deterministic operations ensure reproducibility but may reduce performance slightly
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
