import os
import torch
import numpy as np
import random


def seed_everything(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # --- Experiment Identification ---
    EXP_ID = "idea_16"

    # --- Directory Paths ---
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directory (Outputs)
    WORK_DIR = os.path.join("./working", EXP_ID)
    CACHE_DIR = WORK_DIR
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORK_DIR, "submission")

    # Ensure output directories exist
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- Data Configuration ---
    IMG_SIZE = 32
    NUM_CLASSES = 1  # Binary classification

    # --- Training Hyperparameters ---
    SEED = 42
    N_FOLDS = 5
    EPOCHS = 30
    BATCH_SIZE = 128
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # Optimization
    LEARNING_RATE = 1e-3
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-2

    # Regularization & Augmentation
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2

    # Multi-Task Learning (MTL)
    USE_MTL = True
    AUX_LOSS_WEIGHT = 1.0  # Weight for the File Size Regression Loss

    # Feature-wise Linear Modulation (FiLM)
    USE_FILM = True

    # Stochastic Weight Averaging (SWA)
    USE_SWA = True
    SWA_START_EPOCH = 20
    SWA_LR = 1e-4

    # --- Architecture Selection ---
    # List of backbone architectures to train
    MODELS_TO_RUN = ["RepVGG", "ResNet", "NeXt"]

    # --- Compute ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Debugging ---
    DEBUG = False  # Set to True to run on a small subset for testing
    DEBUG_SUBSET_SIZE = 1000

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print(f"=== Configuration for {cls.EXP_ID} ===")
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}, Epochs: {cls.EPOCHS}")
        print(f"Models: {cls.MODELS_TO_RUN}")
        print(f"Mixup: {cls.USE_MIXUP} (alpha={cls.MIXUP_ALPHA})")
        print(f"MTL: {cls.USE_MTL} (Aux Weight={cls.AUX_LOSS_WEIGHT})")
        print(f"FiLM: {cls.USE_FILM}")
        print(f"SWA: {cls.USE_SWA} (Start={cls.SWA_START_EPOCH})")
        print(f"========================================")


# Apply seed globally upon import
seed_everything(Config.SEED)
