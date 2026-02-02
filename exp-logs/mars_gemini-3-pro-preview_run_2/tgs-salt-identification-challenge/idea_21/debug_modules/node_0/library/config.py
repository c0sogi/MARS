import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
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
    Global configuration class for the Salt Segmentation task.
    Contains paths, hyperparameters, and constants for the
    Privileged Multi-Task Distillation strategy.
    """

    # =========================================================================
    # Paths
    # =========================================================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_21"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Cache directory for deterministic data processing
    CACHE_DIR = WORKING_DIR

    # Model Checkpoint paths
    TEACHER_CHECKPOINT = os.path.join(WORKING_DIR, "teacher_best.pth")
    STUDENT_CHECKPOINT = os.path.join(WORKING_DIR, "student_best.pth")
    FINAL_SUBMISSION_PATH = "./submission/submission.csv"

    # =========================================================================
    # Data & Preprocessing
    # =========================================================================
    ORIG_SIZE = 101
    IMG_SIZE = 128  # Padded size for U-Net/LinkNet divisibility
    NUM_WORKERS = 4

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    ENCODER = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 1  # We sum RGB channels to create 1-channel input

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Phase 1: Teacher Training
    TEACHER_EPOCHS = 50

    # Phase 2: Student Distillation
    STUDENT_EPOCHS = 50

    # Loss Weights for Student (Composite Loss)
    LAMBDA_DISTILL = 1.0  # Weight for distillation loss
    LAMBDA_DEPTH = 1.0  # Weight for auxiliary depth regression loss

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # =========================================================================
    # Augmentation Parameters
    # =========================================================================
    # Elastic Transform (Crucial for salt plasticity)
    AUG_ELASTIC_ALPHA = 120
    AUG_ELASTIC_SIGMA = 6
    AUG_ELASTIC_ALPHA_AFFINE = 120 * 0.03

    # Rigid Transform
    AUG_SHIFT_SCALE_ROTATE_P = 0.2

    # =========================================================================
    # Post-Processing
    # =========================================================================
    # Threshold range for optimization (0.5 to 0.95 step 0.05 is the metric)
    # We search a broader range for the binary mask threshold
    THRESHOLD_SEARCH_START = 0.3
    THRESHOLD_SEARCH_END = 0.7
    THRESHOLD_SEARCH_STEP = 0.01
