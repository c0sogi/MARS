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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Config:
    """
    Configuration class to manage hyperparameters, paths, and runtime settings.
    """

    def __init__(
        self,
        seed=42,
        img_size=256,
        batch_size=64,
        epochs=3,
        learning_rate=1e-3,
        num_workers=4,
        debug=False,
        debug_sample_size=1000,
    ):
        """
        Initialize configuration with default or custom values.

        Args:
            seed (int): Random seed.
            img_size (int): Input image size (square).
            batch_size (int): Batch size for training/inference.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            num_workers (int): Number of dataloader workers.
            debug (bool): If True, limits dataset size for rapid debugging.
            debug_sample_size (int): Number of samples to use when debug is True.
        """
        self.seed = seed
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.debug = debug
        self.debug_sample_size = debug_sample_size

        # Hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Paths
        self.input_dir = "./input"
        self.metadata_dir = "./metadata"
        self.train_metadata = os.path.join(self.metadata_dir, "train.csv")
        self.val_metadata = os.path.join(self.metadata_dir, "val.csv")
        self.test_metadata = os.path.join(self.metadata_dir, "test.csv")

        # Output Paths
        self.working_dir = "./working/idea_1"
        self.model_path = os.path.join(self.working_dir, "model.pth")
        self.submission_dir = "./submission"
        self.submission_path = os.path.join(self.submission_dir, "submission.csv")

        # Create necessary directories
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

    def to_dict(self):
        """Returns configuration as a dictionary."""
        return {
            "seed": self.seed,
            "img_size": self.img_size,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "device": str(self.device),
            "debug": self.debug,
        }
