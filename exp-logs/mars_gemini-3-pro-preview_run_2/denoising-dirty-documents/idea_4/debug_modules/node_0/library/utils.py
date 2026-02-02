import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_rmse(predictions, targets):
    """
    Calculates the Root Mean Squared Error (RMSE) between predictions and targets.

    Args:
        predictions (torch.Tensor or np.ndarray): Predicted pixel intensities.
        targets (torch.Tensor or np.ndarray): Ground truth pixel intensities.

    Returns:
        float: The RMSE value.
    """
    # Convert tensors to numpy if necessary
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs are flattened to treat all pixels equally
    predictions = predictions.flatten()
    targets = targets.flatten()

    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)

    return rmse


def save_checkpoint(state: dict, filename: str = Config.MODEL_SAVE_PATH):
    """
    Saves the model checkpoint to the specified file.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        filename (str): Path to save the checkpoint. Defaults to Config.MODEL_SAVE_PATH.
    """
    # Ensure the directory exists
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filename)
    # Print implicitly handled by training loop, but useful for debugging
    # print(f"Checkpoint saved to {filename}")


def load_checkpoint(
    model,
    filename: str = Config.MODEL_SAVE_PATH,
    optimizer=None,
    scheduler=None,
    device=Config.DEVICE,
):
    """
    Loads a model checkpoint from the specified file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filename (str): Path to the checkpoint file. Defaults to Config.MODEL_SAVE_PATH.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (optional): Scheduler to load state into.
        device (str): Device to map the location to (e.g., 'cuda' or 'cpu').

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch or best score).
        None: If the file does not exist.
    """
    if not os.path.exists(filename):
        print(f"No checkpoint found at {filename}")
        return None

    checkpoint = torch.load(filename, map_location=device)

    # Load model state
    # Handle cases where DataParallel might have been used (keys start with 'module.')
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # Assume raw state dict if key not present

    # Basic handling for DataParallel wrapper removal if current model is not wrapped but checkpoint is
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    print(f"Checkpoint loaded from {filename}")
    return checkpoint
