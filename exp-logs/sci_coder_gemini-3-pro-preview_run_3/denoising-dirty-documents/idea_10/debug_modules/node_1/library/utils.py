import os
import random
import shutil
import numpy as np
import torch
from library import config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_rmse(predictions, targets):
    """
    Calculates the Root Mean Squared Error (RMSE) between predictions and targets.

    Args:
        predictions (np.ndarray or torch.Tensor): Predicted values.
        targets (np.ndarray or torch.Tensor): Ground truth values.

    Returns:
        float: The RMSE value.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Calculate MSE
    mse = np.mean((predictions - targets) ** 2)

    # Calculate RMSE
    rmse = np.sqrt(mse)
    return rmse


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint to the working directory.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, epoch, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Name of the checkpoint file.
    """
    # Ensure the working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    filepath = os.path.join(config.WORKING_DIR, filename)
    torch.save(state, filepath)

    # If this is the best model, save a copy to the best model path
    if is_best:
        shutil.copyfile(filepath, config.BEST_MODEL_PATH)


def load_checkpoint(checkpoint_path, model, optimizer=None, device=config.DEVICE):
    """
    Loads a model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        device (str): Device to map the location to.

    Returns:
        tuple: (start_epoch, best_metric)
            start_epoch (int): The epoch to resume from.
            best_metric (float): The best metric value recorded in the checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        # Return defaults if no checkpoint found
        return 0, float("inf")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["state_dict"])

    # Load optimizer state if provided
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Extract metadata
    start_epoch = checkpoint.get("epoch", 0)
    best_metric = checkpoint.get("best_metric", float("inf"))

    return start_epoch, best_metric


def normalize_image(image):
    """
    Normalizes image pixel intensities from [0, 255] to [0, 1].

    Args:
        image (np.ndarray): Input image array (uint8 or float).

    Returns:
        np.ndarray: Normalized image array (float32).
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image):
    """
    Denormalizes image pixel intensities from [0, 1] to [0, 255].

    Args:
        image (np.ndarray): Normalized image array.

    Returns:
        np.ndarray: Denormalized image array (uint8).
    """
    # Clip values to valid range
    image = np.clip(image, 0.0, 1.0)
    # Scale and convert to uint8
    return (image * 255.0).astype(np.uint8)


def count_parameters(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
