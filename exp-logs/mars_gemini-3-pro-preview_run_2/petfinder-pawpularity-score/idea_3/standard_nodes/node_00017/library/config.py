import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for standard python, numpy, and torch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Deterministic behavior is slower but required for exact reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    """
    Configuration class for the Pet Pawpularity Prediction task.
    Holds hyperparameters for data, model, training, and paths.
    """

    # ------------------
    # General Settings
    # ------------------
    seed = 42
    debug = False  # Set to True to run on a small subset of data
    debug_sample_size = 100

    # ------------------
    # Compute
    # ------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4  # Number of subprocesses for data loading

    # ------------------
    # Paths
    # ------------------
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working/idea_3"
    submission_dir = "./submission"

    # Metadata file paths
    train_csv_path = os.path.join(metadata_dir, "train.csv")
    val_csv_path = os.path.join(metadata_dir, "validation.csv")
    test_csv_path = os.path.join(metadata_dir, "test.csv")
    submission_path = os.path.join(submission_dir, "submission.csv")

    # ------------------
    # Data Parameters
    # ------------------
    image_size = 384
    batch_size = 8  # Reduced batch size for 384x384 resolution
    target_col = "Pawpularity"

    # ------------------
    # Model Architecture
    # ------------------
    # Heterogeneous ensemble backbones
    model_names = ["swin_large_patch4_window12_384", "convnext_large_384_in22ft1k"]

    # Head parameters
    fc_dim = 128
    dropout = 0.0

    # ------------------
    # Training Hyperparameters
    # ------------------
    num_folds = 5
    epochs = 10

    # Optimization
    lr = 1e-4  # Learning rate for the head
    backbone_lr = 1e-5  # Lower learning rate for the pre-trained backbone
    weight_decay = 1e-6
    min_lr = 1e-6

    # Scheduler
    scheduler_type = "cosine_warmup"
    warmup_epochs = 1

    # Loss
    # We use BCEWithLogitsLoss, so targets must be scaled to [0, 1]

    def __init__(self, **kwargs):
        """
        Initialize Config with optional overrides.
        """
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @classmethod
    def create_dirs(cls):
        """
        Creates necessary working directories.
        """
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)

    def print_config(self):
        """
        Prints the current configuration.
        """
        print("=" * 30)
        print("CONFIGURATION")
        print("=" * 30)
        for k, v in self.__class__.__dict__.items():
            if not k.startswith("__") and not callable(v):
                # Check if instance variable overrides class variable
                val = getattr(self, k)
                print(f"{k}: {val}")
        print("=" * 30)
