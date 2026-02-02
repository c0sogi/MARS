import os
import random
import shutil
import numpy as np
import torch


class Config:
    """
    Global configuration for the Lightweight Metric Learning Baseline.
    """

    # Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Data Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Model Architecture
    BACKBONE = "mobilenet_v3_large"
    EMBEDDING_DIM = 512
    NUM_CLASSES = 7770  # Derived from EDA (Total classes in training)

    # Hyperparameters
    SEED = 42
    IMG_SIZE = 224
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 10

    # ArcFace Hyperparameters
    MARGIN = 0.50
    SCALE = 30.0

    # Output Paths
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "checkpoint.pth")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    @staticmethod
    def setup_dirs():
        """Creates necessary working directories."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self, name="Metric", fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def set_seed(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(state, is_best, filepath):
    """
    Saves the model checkpoint. If is_best is True, copies to best_model.pth.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(directory, "best_model.pth")
        shutil.copyfile(filepath, best_path)


def get_device():
    """
    Returns the appropriate torch device (CUDA or CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
