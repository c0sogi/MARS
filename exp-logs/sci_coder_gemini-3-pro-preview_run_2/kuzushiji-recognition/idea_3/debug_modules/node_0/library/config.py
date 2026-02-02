import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for Kuzushiji Character Recognition.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    def __init__(self, debug=False, num_epochs=15):
        # =========================================================================
        # Directories and Paths
        # =========================================================================
        self.INPUT_DIR = "./input"
        self.METADATA_DIR = "./metadata"
        self.WORKING_DIR = "./working/idea_3"
        self.SUBMISSION_DIR = "./submission"

        # Ensure working and output directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        # Data Paths
        self.TRAIN_CSV = os.path.join(self.METADATA_DIR, "train.csv")
        self.VAL_CSV = os.path.join(self.METADATA_DIR, "val.csv")
        self.TEST_CSV = os.path.join(self.METADATA_DIR, "test.csv")
        self.UNICODE_MAP = os.path.join(self.INPUT_DIR, "unicode_translation.csv")

        # Output Paths
        self.MODEL_PATH = os.path.join(self.WORKING_DIR, "best_model.pth")
        self.SUBMISSION_PATH = os.path.join(self.SUBMISSION_DIR, "submission.csv")

        # =========================================================================
        # Model Configuration
        # =========================================================================
        # 3848 characters + 1 background class
        self.NUM_CLASSES = 3849

        # Input Resolution (High resolution for small characters)
        self.MIN_SIZE = 1024
        self.MAX_SIZE = 2048

        # =========================================================================
        # Training Hyperparameters
        # =========================================================================
        self.BATCH_SIZE = 4
        self.LEARNING_RATE = 0.005
        self.MOMENTUM = 0.9
        self.WEIGHT_DECAY = 0.0001

        # Scheduler
        self.NUM_EPOCHS = num_epochs
        self.LR_STEPS = [10, 13]
        self.LR_GAMMA = 0.1

        # =========================================================================
        # Inference Hyperparameters
        # =========================================================================
        # Keep more proposals to handle dense text
        self.RPN_POST_NMS_TOP_N_TEST = 2000
        # Cap detections to match dataset density
        self.BOX_DETECTIONS_PER_IMG = 1200
        # Lower threshold to improve recall on faint characters
        self.SCORE_THRESHOLD = 0.35

        # =========================================================================
        # System & Debugging
        # =========================================================================
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NUM_WORKERS = 4
        self.SEED = 42

        self.DEBUG = debug
        if self.DEBUG:
            self.NUM_EPOCHS = 1
            self.TRAIN_SAMPLE_SIZE = 100
            self.VAL_SAMPLE_SIZE = 20
        else:
            self.TRAIN_SAMPLE_SIZE = None
            self.VAL_SAMPLE_SIZE = None


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
