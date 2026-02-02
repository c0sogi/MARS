import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Ensures deterministic behavior for CuDNN backend.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_roc_auc(y_true, y_scores):
    """
    Calculates the Area Under the ROC Curve.
    Handles both numpy arrays and torch tensors.

    Args:
        y_true: Ground truth binary labels (shape: [N] or [N, 1]).
        y_scores: Predicted probabilities (shape: [N] or [N, 1]).

    Returns:
        float: ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.detach().cpu().numpy()

    # Flatten arrays to ensure 1D
    y_true = np.ravel(y_true)
    y_scores = np.ravel(y_scores)

    try:
        score = roc_auc_score(y_true, y_scores)
    except ValueError:
        # Handle edge cases where only one class is present in the batch
        # This can happen with small batch sizes or highly imbalanced batches
        score = 0.5

    return score


def save_model(model, seed, filename=None):
    """
    Saves the model state dictionary to the working directory.

    Args:
        model: The PyTorch model instance.
        seed (int): The seed associated with this model run.
        filename (str, optional): Custom filename. If None, defaults to 'model_seed_{seed}.pth'.
    """
    if filename is None:
        filename = f"model_seed_{seed}.pth"

    save_path = os.path.join(Config.WORKING_DIR, filename)
    torch.save(model.state_dict(), save_path)


def load_model(model, seed, filename=None, device=None):
    """
    Loads the model state dictionary from the working directory.

    Args:
        model: The PyTorch model instance to load weights into.
        seed (int): The seed associated with the file to load.
        filename (str, optional): Custom filename. If None, defaults to 'model_seed_{seed}.pth'.
        device (torch.device, optional): Device to map the location to.

    Returns:
        model: The model with loaded weights.
    """
    if filename is None:
        filename = f"model_seed_{seed}.pth"

    load_path = os.path.join(Config.WORKING_DIR, filename)

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file not found at {load_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict)

    return model


class ModelCheckpoint:
    """
    Handles checkpointing logic. Tracks the best metric score and saves the model
    only when the score improves.
    """

    def __init__(self, seed, mode="max"):
        """
        Args:
            seed (int): The random seed for the current training run.
            mode (str): 'max' for metrics like AUC/Accuracy, 'min' for Loss.
        """
        self.seed = seed
        self.mode = mode
        if mode == "max":
            self.best_score = -float("inf")
        else:
            self.best_score = float("inf")

    def step(self, score, model):
        """
        Updates the tracker with a new score and saves the model if it is the best so far.

        Args:
            score (float): The current metric score.
            model: The model to save.

        Returns:
            bool: True if the model was saved (new best), False otherwise.
        """
        if self.mode == "max":
            is_best = score > self.best_score
        else:
            is_best = score < self.best_score

        if is_best:
            self.best_score = score
            save_model(model, self.seed)
            return True
        return False


def print_metrics(epoch, train_loss, val_loss, val_auc):
    """
    Prints training metrics with full precision as required.

    Args:
        epoch (int): Current epoch number.
        train_loss (float): Training loss.
        val_loss (float): Validation loss.
        val_auc (float): Validation AUC score.
    """
    print(
        f"Epoch {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
    )
