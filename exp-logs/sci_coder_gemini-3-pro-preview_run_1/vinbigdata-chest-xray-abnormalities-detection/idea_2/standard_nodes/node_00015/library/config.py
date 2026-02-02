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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for intermediate files (checkpoints, cache)
    WORKING_DIR = "./working/idea_2"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")
    SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure submission directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Data Configuration
    # =========================================================================
    IMG_SIZE = 512  # Input image resolution (Height, Width)
    NUM_CLASSES = 14  # 0-13 are specific findings

    # Class mapping for reference and visualization
    CLASS_ID_TO_NAME = {
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

    # =========================================================================
    # Model Configuration
    # =========================================================================
    BACKBONE = "tf_efficientnet_b0_ns"  # Timm backbone name
    PRETRAINED = True

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 8  # Adjusted for A100 40GB
    NUM_EPOCHS = 15  # Max epochs
    LEARNING_RATE = 3e-4  # Initial learning rate
    NUM_WORKERS = 8  # Number of dataloader workers

    # Optimizer settings
    WEIGHT_DECAY = 1e-4
    CLIP_GRAD_NORM = 1.0

    # =========================================================================
    # Debugging / Development
    # =========================================================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SIZE = 500  # Number of samples to use in debug mode

    # =========================================================================
    # Inference / Post-processing
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Thresholds
    CONF_THRESHOLD = 0.15  # Minimum confidence for bounding boxes
    IOU_THRESHOLD = 0.4  # IoU threshold for evaluation metric

    # Global Classification Head Threshold
    # If the model predicts "No Finding" probability > this, output class 14
    NO_FINDING_THRESH = 0.70
