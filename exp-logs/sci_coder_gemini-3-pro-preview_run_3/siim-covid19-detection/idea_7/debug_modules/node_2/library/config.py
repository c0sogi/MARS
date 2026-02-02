import os
import torch
import random
import numpy as np


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
    # =========================
    # General Settings
    # =========================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = os.cpu_count()  # Utilize all available vCPUs

    # Debugging
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True

    # =========================
    # Directories & Paths
    # =========================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    WORKING_DIR = "./working/idea_7"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = WORKING_DIR
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================
    # Data Configuration
    # =========================
    IMG_SIZE = 800  # Target size for Letterbox Resizing (Longest Edge)
    RESIZE_MODE = "letterbox"

    # Class Definitions
    # Study Level: 4 Classes
    STUDY_LABELS = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]
    NUM_STUDY_CLASSES = len(STUDY_LABELS)

    # Detection Level: 1 Class (+ Background implicit in R-CNN)
    # PyTorch Faster/Cascade R-CNN expects num_classes to include background (index 0)
    DETECTION_LABELS = ["opacity"]
    NUM_DETECTION_CLASSES = len(DETECTION_LABELS) + 1  # 1 background + 1 opacity = 2

    # =========================
    # Model Configuration
    # =========================
    MODEL_NAME = "cascade_rcnn_convnextv2"
    BACKBONE = "convnextv2_base"  # timm backbone
    PRETRAINED = True

    # Cascade R-CNN Hyperparameters
    # IoU thresholds for the 3 stages of Cascade R-CNN
    CASCADE_IOU_THRESHOLDS = [0.5, 0.6, 0.7]

    # RPN Settings
    RPN_PRE_NMS_TOP_N_TRAIN = 2000
    RPN_POST_NMS_TOP_N_TRAIN = 2000
    RPN_PRE_NMS_TOP_N_TEST = 1000
    RPN_POST_NMS_TOP_N_TEST = 1000
    RPN_NMS_THRESH = 0.7

    # ROI Heads Settings
    BOX_SCORE_THRESH = 0.05
    BOX_NMS_THRESH = 0.5
    BOX_DETECTIONS_PER_IMG = 100

    # =========================
    # Training Configuration
    # =========================
    BATCH_SIZE = 8  # Adjusted for A100 GPU and Model Size
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.05  # AdamW weight decay
    MAX_GRAD_NORM = 10.0

    # Learning Rate Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Loss Weights (Multi-task)
    LOSS_WEIGHT_DETECTION = 1.0
    LOSS_WEIGHT_STUDY = 1.0

    # =========================
    # Inference & Post-Processing
    # =========================
    # Test Time Augmentation
    USE_TTA = True  # Horizontal Flip

    # Weighted Boxes Fusion (WBF)
    WBF_IOU_THRESHOLD = 0.5
    WBF_CONF_THRESHOLD = 0.001  # Keep low confidence boxes for fusion

    # Final Prediction Logic
    # If study prediction is "Negative", force "none 1 0 0 1 1"
    NEGATIVE_CLASS_IDX = 0  # Index of 'Negative for Pneumonia' in STUDY_LABELS


# Initialize seed on import
seed_everything(Config.SEED)
