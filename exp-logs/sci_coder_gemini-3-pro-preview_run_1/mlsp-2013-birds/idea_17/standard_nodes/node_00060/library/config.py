import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
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
    """
    Global configuration for the Multi-Generational Noisy Student Distillation pipeline.
    """

    # ==========================================
    #                General
    # ==========================================
    SEED = 42
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50  # Number of samples to use when DEBUG is True
    NUM_WORKERS = 2  # Data loading workers

    # ==========================================
    #                 Paths
    # ==========================================
    # Input Data
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Image Source
    SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")

    # Working Directory (Idea 17)
    IDEA_NAME = "idea_17"
    WORK_DIR = os.path.join("./working", IDEA_NAME)

    # Sub-directories for artifacts
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    LOG_DIR = os.path.join(WORK_DIR, "logs")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    #              Model Architecture
    # ==========================================
    BACKBONE = "resnet34"
    NUM_CLASSES = 19
    PRETRAINED = True

    # ==========================================
    #           Data Preprocessing
    # ==========================================
    # High-Fidelity Resolution
    IMG_HEIGHT = 256
    IMG_WIDTH = 640
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

    # ImageNet Normalization Statistics
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ==========================================
    #           Training Hyperparameters
    # ==========================================
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Stochastic Weight Averaging (SWA)
    # Active for final 40% of epochs (approx last 20 epochs)
    SWA_START_EPOCH = 30
    SWA_LR = 1e-4

    # Regularization
    DROPOUT_P = 0.5  # Probability for Head Dropout
    MIXUP_ALPHA = 0.2  # Alpha for Beta distribution in Mixup

    # ==========================================
    #           Pipeline Stages
    # ==========================================
    NUM_TEACHERS = 3  # Number of models in the Teacher Ensemble

    @classmethod
    def setup(cls):
        """
        Initialize the workspace: create directories and set seeds.
        """
        # Create necessary directories
        for d in [
            cls.WORK_DIR,
            cls.CHECKPOINT_DIR,
            cls.CACHE_DIR,
            cls.LOG_DIR,
            cls.SUBMISSION_DIR,
        ]:
            os.makedirs(d, exist_ok=True)

        # Set deterministic seed
        set_seed(cls.SEED)
