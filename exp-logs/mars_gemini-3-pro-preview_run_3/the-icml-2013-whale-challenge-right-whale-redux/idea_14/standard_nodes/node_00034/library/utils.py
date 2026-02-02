import os
import random
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: dict, filepath: str) -> None:
    """
    Saves the model checkpoint to the specified file path.

    Args:
        state (dict): The state dictionary containing model, optimizer, and metadata.
        filepath (str): The full path where the checkpoint will be saved.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filepath)


def load_checkpoint(
    model: nn.Module,
    filepath: str,
    optimizer: torch.optim.Optimizer = None,
    device: str = "cpu",
) -> dict:
    """
    Loads a checkpoint into the model and optionally the optimizer.

    Args:
        model (nn.Module): The model to load weights into.
        filepath (str): The path to the checkpoint file.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into. Defaults to None.
        device (str): The device to map the checkpoint to ('cpu' or 'cuda').

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch or metrics).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    # Handle cases where the model might be wrapped in DataParallel (keys starting with 'module.')
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    # If the current model is not DataParallel but the checkpoint is, remove 'module.' prefix
    if not isinstance(model, nn.DataParallel):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def mixup_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4, device: str = "cuda"
):
    """
    Performs Mixup augmentation on the input batch.

    Args:
        x (torch.Tensor): Input batch of data.
        y (torch.Tensor): Target labels.
        alpha (float): The alpha parameter for the Beta distribution.
        device (str): The device to perform operations on.

    Returns:
        tuple: (mixed_x, y_a, y_b, lam)
            - mixed_x: The mixed input data.
            - y_a: Targets for the first component.
            - y_b: Targets for the second component.
            - lam: The mixing coefficient lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: callable,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Computes the Mixup loss given predictions and two sets of targets.

    Args:
        criterion (callable): The loss function (e.g., nn.BCEWithLogitsLoss).
        pred (torch.Tensor): The model predictions.
        y_a (torch.Tensor): The first set of targets.
        y_b (torch.Tensor): The second set of targets (permuted).
        lam (float): The mixing coefficient.

    Returns:
        torch.Tensor: The combined loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
