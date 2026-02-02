import os
import random
import numpy as np
import torch
from sklearn.metrics import log_loss
from torch.optim.swa_utils import AveragedModel, update_bn
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metric(y_true, y_pred):
    """
    Calculates the Multi Class Log Loss.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth labels (indices).
        y_pred (torch.Tensor or np.ndarray): Predicted probabilities (N x C).

    Returns:
        float: The calculated log loss.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Explicitly provide labels to ensure log_loss handles missing classes in the batch correctly
    labels = list(range(Config.NUM_CLASSES))

    return log_loss(y_true, y_pred, labels=labels)


class SWAHandler:
    """
    Manages Stochastic Weight Averaging (SWA) model collection and buffer updates.
    Wraps torch.optim.swa_utils.AveragedModel.
    """

    def __init__(self, model, device):
        """
        Args:
            model (torch.nn.Module): The base model to average.
            device (str or torch.device): The device to keep the averaged model on.
        """
        self.device = device
        self.swa_model = AveragedModel(model).to(device)

    def update(self, model):
        """
        Updates the averaged model parameters with the current model parameters.

        Args:
            model (torch.nn.Module): The current model state.
        """
        self.swa_model.update_parameters(model)

    def update_bn(self, train_loader):
        """
        Updates Batch Normalization statistics for the averaged model using the training data.

        Args:
            train_loader (DataLoader): DataLoader to compute BN stats.
        """
        update_bn(train_loader, self.swa_model, device=self.device)

    def get_model(self):
        """
        Returns the averaged model.

        Returns:
            torch.nn.Module: The SWA model.
        """
        return self.swa_model


class Logger:
    """
    Simple logger to track training progress in a file and on the console.
    """

    def __init__(self, log_file):
        """
        Args:
            log_file (str): Path to the log file.
        """
        self.log_file = log_file
        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Initialize/Clear the log file
        with open(self.log_file, "w") as f:
            f.write(f"Training Log initialized at {self.log_file}\n")

    def log(self, message):
        """
        Prints message to stdout and appends to the log file.

        Args:
            message (str): The message to log.
        """
        print(message)
        with open(self.log_file, "a") as f:
            f.write(str(message) + "\n")
