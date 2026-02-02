import os
import torch


def count_parameters(model):
    """
    Counts the number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        loss (float): The validation loss at this epoch.
        path (str): The file path to save the checkpoint.
    """
    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """
    Loads the model and optimizer state from a checkpoint file.

    Args:
        path (str): The file path to load the checkpoint from.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str or torch.device): The device to map the location to.

    Returns:
        dict: The checkpoint dictionary containing epoch and loss, or None if not found.
    """
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
