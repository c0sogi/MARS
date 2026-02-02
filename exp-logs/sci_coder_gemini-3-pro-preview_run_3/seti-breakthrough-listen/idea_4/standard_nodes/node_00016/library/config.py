import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the Siamese EfficientNet-B0 Technosignature Detection model.
    """

    # --- File Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory specific to Idea 4 (Siamese Network)
    WORK_DIR = "./working/idea_4"

    # Metadata CSV paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Submission paths
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- System Settings ---
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Available vCPUs: 12. Using a safe number for workers.
    NUM_WORKERS = 8

    # --- Model Hyperparameters ---
    MODEL_NAME = "efficientnet_b0"
    # Input image size: (Height, Width)
    # We resize to (256, 256) to stay closer to native (273, 256) and preserve details
    # Cite solution_lesson_node_00004
    IMG_SIZE = (256, 256)
    # Number of input channels per stream (3 On-target, 3 Off-target)
    IN_CHANNELS = 3
    NUM_CLASSES = 1

    # --- Training Hyperparameters ---
    BATCH_SIZE = 64
    EPOCHS = 12
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MIXUP_ALPHA = 0.2

    # Scheduler settings (CosineAnnealingLR)
    T_MAX = 10

    # --- Debugging ---
    # Toggle to True to train on a small subset
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000

    @staticmethod
    def setup_directories():
        """Creates necessary working and submission directories."""
        os.makedirs(Config.WORK_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    @staticmethod
    def seed_everything(seed: int = 42):
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


# Initialize environment configuration
Config.setup_directories()
Config.seed_everything(Config.SEED)
