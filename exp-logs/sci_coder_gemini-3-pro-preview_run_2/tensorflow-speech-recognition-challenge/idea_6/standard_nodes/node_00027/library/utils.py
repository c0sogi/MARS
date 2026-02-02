import os
import random
import numpy as np
import torch
from library import config


def set_seed(seed=config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, is_best, filename=config.MODEL_CHECKPOINT_PATH):
    """
    Saves the model checkpoint.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): The path to save the checkpoint to.
    """
    # We only save if it is the best model to save disk space and IO time,
    # as per the strategy to keep the best submission.
    if is_best:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)


def load_checkpoint(model, filename=config.MODEL_CHECKPOINT_PATH, device=config.DEVICE):
    """
    Loads the model weights from a checkpoint file.

    Args:
        model (torch.nn.Module): The model architecture to load weights into.
        filename (str): The path to the checkpoint file.
        device (str): The device to map the location to.

    Returns:
        tuple: (model, best_acc) where best_acc is the accuracy stored in the checkpoint.
               If file not found, returns (model, 0.0).
    """
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=device)
        # Handle cases where the state_dict might be nested or just the dict itself
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            best_acc = checkpoint.get("best_acc", 0.0)
        else:
            model.load_state_dict(checkpoint)
            best_acc = 0.0
        return model, best_acc
    return model, 0.0


def log_metrics(epoch, train_loss, val_loss, val_acc, time_elapsed):
    """
    Prints training and validation metrics with full precision.

    Args:
        epoch (int): The current epoch number.
        train_loss (float): The training loss.
        val_loss (float): The validation loss.
        val_acc (float): The validation accuracy.
        time_elapsed (float): Time taken for the epoch in seconds.
    """
    print(
        f"Epoch {epoch} | Time: {time_elapsed}s | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Acc: {val_acc}"
    )
