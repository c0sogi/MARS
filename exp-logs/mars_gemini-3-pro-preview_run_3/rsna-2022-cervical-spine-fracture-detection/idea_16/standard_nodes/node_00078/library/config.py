import os
import torch
import random
import numpy as np


class Config:
    # ====================================================
    # General Settings
    # ====================================================
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use in debug mode

    # ====================================================
    # Directories & Paths
    # ====================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Caching Directory (Required by Idea 16)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_16")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data Paths
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ====================================================
    # Data Preprocessing (Idea: 2.5D Stacks, Bone Window)
    # ====================================================
    IMAGE_SIZE = 224  # Resolution: 224x224
    NUM_SLICES = 64  # Uniform sampling of 64 slices
    CHANNELS = 3  # 2.5D Stacking (z-1, z, z+1)

    # Standard Bone Window Settings
    BONE_WINDOW_LEVEL = 400
    BONE_WINDOW_WIDTH = 1800

    # ====================================================
    # Model Architecture (Idea: ConvNeXt-Tiny + MIL)
    # ====================================================
    BACKBONE = "convnext_tiny.in12k_ft_in1k"  # timm model name
    PRETRAINED = True
    IN_CHANNELS = 3
    NUM_CLASSES = 8  # 7 vertebrae + 1 patient_overall

    # ====================================================
    # Training Hyperparameters
    # ====================================================
    EPOCHS = 10
    BATCH_SIZE = 8  # Small batch size for stability with LayerNorm
    NUM_WORKERS = 4  # Number of dataloader workers

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1000.0

    # Scheduler (Decoupled Cosine Annealing)
    # T_max is set dynamically based on epochs in training loop,
    # but multiplier is defined here.
    T_MAX_MULT = 1.5  # 1.5x the number of epochs
    MIN_LR = 1e-6

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Apply seeding immediately upon import
seed_everything(Config.SEED)
