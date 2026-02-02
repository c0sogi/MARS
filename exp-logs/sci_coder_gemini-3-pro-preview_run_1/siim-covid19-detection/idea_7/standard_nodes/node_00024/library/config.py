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
    """
    Centralized configuration for the Deeply Supervised ResNet18 Multi-Task U-Net.
    """

    # ==========================
    # General Settings
    # ==========================
    PROJECT_NAME = "siim-covid19-detection"
    EXPERIMENT_NAME = "idea_7_deep_sup_resnet18"
    SEED = 42

    # ==========================
    # Directories & Paths
    # ==========================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata CSV Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Model Checkpoint Path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================
    # Data Parameters
    # ==========================
    IMG_SIZE = 512
    NUM_WORKERS = 12  # Optimized for 12 vCPUs
    BATCH_SIZE = 32  # Max feasible for A100 with ResNet18

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200  # Number of images to use when DEBUG is True

    # ==========================
    # Model Architecture
    # ==========================
    BACKBONE = "resnet18"
    PRETRAINED = True
    NUM_STUDY_CLASSES = 4

    # Study Label Mapping (Order matters for one-hot encoding/prediction)
    STUDY_LABELS = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]

    # ==========================
    # Training Hyperparameters
    # ==========================
    EPOCHS = 20
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-5

    # Optimizer & Scheduler
    OPTIMIZER = "AdamW"
    SCHEDULER = "CosineAnnealingLR"
    MIN_LR = 1e-6

    # ==========================
    # Loss Weights
    # ==========================
    # Global balance: L_total = CLS_WEIGHT * L_cls + SEG_WEIGHT * L_seg
    CLS_LOSS_WEIGHT = 1.0
    SEG_LOSS_WEIGHT = 10.0

    # Deep Supervision Weights for Segmentation Heads
    # L_seg = weights[0]*Final + weights[1]*Aux1(1/2) + weights[2]*Aux2(1/4)
    DEEP_SUP_WEIGHTS = [1.0, 0.5, 0.25]

    # ==========================
    # Inference
    # ==========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Thresholds
    CONF_THRESHOLD = 0.001  # Low threshold for raw predictions, filtered later
    IOU_THRESHOLD = 0.5  # For mAP calculation

    @classmethod
    def get_summary(cls):
        """Returns a dictionary of the current configuration."""
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and not callable(v)
        }
