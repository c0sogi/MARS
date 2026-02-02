import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # System & Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of dataloader workers

    # ==========================================
    # File Paths
    # ==========================================
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_27"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Data Sources
    SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")

    # Output Paths
    SUBMISSION_PATH = "./submission/submission.csv"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Cache Paths
    PSEUDO_LABEL_PATH = os.path.join(WORKING_DIR, "pseudo_labels.parquet")

    # ==========================================
    # Data Parameters
    # ==========================================
    # High-Fidelity Resolution (Strictly enforced)
    IMG_HEIGHT = 256
    IMG_WIDTH = 640

    # Normalization (ImageNet stats)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    NUM_CLASSES = 19

    # Augmentation
    MIXUP_ALPHA = 0.2

    # ==========================================
    # Model Parameters
    # ==========================================
    MODEL_NAME = "resnet34"
    PRETRAINED = True
    DROPOUT = 0.0  # Explicitly 0 as per lessons learned

    # ==========================================
    # Training Parameters
    # ==========================================
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # SWA (Stochastic Weight Averaging) Schedule
    # Teacher: Active for final ~25% (Epoch 38+)
    SWA_START_EPOCH_TEACHER = 38
    # Student: Active for final ~30% (Epoch 35+)
    SWA_START_EPOCH_STUDENT = 35
    SWA_LR = 1e-4

    # ==========================================
    # Distillation Parameters
    # ==========================================
    TEACHER_TEMP = 1.5  # Temperature for soft targets
    NUM_TEACHERS = 3  # Number of teacher models in ensemble


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
