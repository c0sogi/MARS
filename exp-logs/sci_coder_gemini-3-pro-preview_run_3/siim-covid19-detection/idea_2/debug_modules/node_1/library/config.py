import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Config:
    # =========================================================================
    # General Configuration
    # =========================================================================
    SEED = 42
    DEBUG = False
    NUM_WORKERS = 8  # Optimized for 12 vCPUs available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache Paths (For deterministic data processing)
    CACHED_TRAIN_DF_PATH = os.path.join(WORKING_DIR, "cached_train_df.parquet")
    CACHED_VAL_DF_PATH = os.path.join(WORKING_DIR, "cached_val_df.parquet")
    CACHED_TEST_DF_PATH = os.path.join(WORKING_DIR, "cached_test_df.parquet")

    # Output Paths
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # =========================================================================
    # Model & Training Hyperparameters
    # =========================================================================
    MODEL_ARCH = "yolov8n"  # YOLOv8-Nano
    IMAGE_SIZE = 640  # 640x640 resolution
    BATCH_SIZE = 32  # Efficient for A100 GPU with Nano model
    EPOCHS = 15  # Constrained for runtime
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-4

    # =========================================================================
    # Class Mapping & Labels
    # =========================================================================
    # Specific class mapping as requested:
    # Typical: 1, Indeterminate: 2, Atypical: 3
    # Note: 'Negative for Pneumonia' is treated as background (no bounding box).
    CLASS_MAPPING = {
        "Typical Appearance": 1,
        "Indeterminate Appearance": 2,
        "Atypical Appearance": 3,
    }

    # Reverse mapping for inference lookup
    ID_TO_CLASS = {v: k for k, v in CLASS_MAPPING.items()}

    # Number of detection classes (excluding background if handled implicitly by YOLO)
    NUM_CLASSES = 3

    # Submission String Templates
    NEGATIVE_STUDY_LABEL = "negative"
    TYPICAL_STUDY_LABEL = "typical"
    INDETERMINATE_STUDY_LABEL = "indeterminate"
    ATYPICAL_STUDY_LABEL = "atypical"

    # Map ID to Submission Study Label
    ID_TO_SUBMISSION_STRING = {
        1: TYPICAL_STUDY_LABEL,
        2: INDETERMINATE_STUDY_LABEL,
        3: ATYPICAL_STUDY_LABEL,
    }

    # Image-level prediction strings
    OPACITY_LABEL = "opacity"
    NONE_PREDICTION = "none 1 0 0 1 1"

    # =========================================================================
    # Inference Constants
    # =========================================================================
    CONF_THRESHOLD = 0.25  # Confidence threshold for detection
    IOU_THRESHOLD = 0.5  # IoU threshold for NMS and mAP calculation
