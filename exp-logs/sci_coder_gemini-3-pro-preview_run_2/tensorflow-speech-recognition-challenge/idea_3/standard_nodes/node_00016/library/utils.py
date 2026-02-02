import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior in CUDNN backends.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, val_acc, path=Config.BEST_MODEL_PATH):
    """
    Saves the model and optimizer state to a checkpoint file.
    Ensures the directory exists before saving.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model, optimizer=None, path=Config.BEST_MODEL_PATH, device=Config.DEVICE
):
    """
    Loads the model and optimizer state from a checkpoint file.
    Returns a dictionary containing metadata (epoch, val_acc) if successful, else None.
    """
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {"epoch": checkpoint.get("epoch"), "val_acc": checkpoint.get("val_acc")}


def calculate_accuracy(outputs, targets):
    """
    Calculates multiclass accuracy.

    Args:
        outputs (torch.Tensor): Logits or probabilities of shape (batch_size, num_classes).
        targets (torch.Tensor): Ground truth labels of shape (batch_size).

    Returns:
        float: The accuracy score between 0.0 and 1.0.
    """
    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)

        if total == 0:
            return 0.0

        return correct / total


def print_metrics(phase, epoch, loss, accuracy):
    """
    Prints training or validation metrics with full precision.
    """
    print(f"Phase: {phase} | Epoch: {epoch} | Loss: {loss} | Accuracy: {accuracy}")
