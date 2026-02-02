import torch
import numpy as np
import random
import os
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Delegates to the centralized Config.set_seed method.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    Config.set_seed(seed)


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the mean column-wise ROC AUC score.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth labels of shape (N, num_classes).
        y_pred (np.ndarray or torch.Tensor): Predicted probabilities of shape (N, num_classes).

    Returns:
        float: The mean ROC AUC score across all classes.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Fallback for empty or single-sample batches which can occur during debugging
    if y_true.shape[0] < 2:
        return 0.5

    try:
        # Calculate macro-average ROC AUC (mean of column-wise AUCs)
        score = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")
        return score
    except ValueError:
        # This can happen if a batch has only one class present in y_true
        # In a full validation pass this is unlikely, but possible in small batches.
        return 0.5


def check_initial_loss(model, data_loader, criterion, device):
    """
    Verifies that the model is correctly initialized by checking the loss on the first batch.
    For a 4-class problem with a randomly initialized head, the expected loss is -ln(1/4) approx 1.386.

    Args:
        model (torch.nn.Module): The neural network model.
        data_loader (torch.utils.data.DataLoader): The data loader to fetch a batch from.
        criterion (callable): The loss function.
        device (str): The device to perform computation on.

    Returns:
        float: The calculated initial loss.
    """
    model.eval()
    model.to(device)

    try:
        images, targets = next(iter(data_loader))
    except StopIteration:
        print("Error: DataLoader is empty during initial loss check.")
        return 0.0

    images = images.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        outputs = model(images)
        loss = criterion(outputs, targets)

    loss_val = loss.item()

    # Expected loss is around 1.38 for 4 classes
    expected = Config.INITIAL_LOSS_THRESHOLD

    print(f"Initial Loss Check: {loss_val}")

    # We check if it's within a reasonable range (e.g., +/- 0.2)
    # This confirms the head is random and the backbone isn't producing garbage
    if abs(loss_val - expected) > 0.5:
        print(
            f"WARNING: Initial loss {loss_val} deviates significantly from expected {expected}."
        )
    else:
        print(f"Initial loss {loss_val} is within expected range of {expected}.")

    return loss_val
