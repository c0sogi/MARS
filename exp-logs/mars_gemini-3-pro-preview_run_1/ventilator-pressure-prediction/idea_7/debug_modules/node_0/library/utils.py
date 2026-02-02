import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Determines and returns the available computing device.

    Returns:
        torch.device: 'cuda' if available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(state: dict, filename: str):
    """
    Saves the training checkpoint to the specified file.

    Args:
        state (dict): A dictionary containing model state, optimizer state, etc.
        filename (str): The path where the checkpoint will be saved.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(
    filename: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    device: torch.device = None,
):
    """
    Loads a checkpoint into the model, optimizer, and scheduler.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (optional): The scheduler to load state into.
        device (torch.device, optional): The device to map the storage to.

    Returns:
        tuple: (start_epoch, best_loss) retrieved from the checkpoint.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    if device is None:
        device = get_device()

    # Load checkpoint mapping to the correct device
    checkpoint = torch.load(filename, map_location=device)

    # Load model state
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)  # Fallback if only state_dict was saved

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0)
    best_loss = checkpoint.get("best_loss", float("inf"))

    return start_epoch, best_loss


def compute_metric(y_pred, y_true, u_out):
    """
    Computes the Mean Absolute Error (MAE) specifically for the inspiratory phase.
    The metric is only calculated where u_out == 0.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted pressure values.
        y_true (torch.Tensor or np.ndarray): Ground truth pressure values.
        u_out (torch.Tensor or np.ndarray): Control input indicating phase (0=Inspiratory, 1=Expiratory).

    Returns:
        float: The MAE for the inspiratory phase.
    """
    # Convert to torch tensors if inputs are numpy arrays
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(u_out, torch.Tensor):
        u_out = torch.tensor(u_out)

    # Ensure all tensors are on the same device (using y_pred as reference)
    device = y_pred.device
    if y_true.device != device:
        y_true = y_true.to(device)
    if u_out.device != device:
        u_out = u_out.to(device)

    # Flatten the tensors to 1D arrays
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    u_out = u_out.view(-1)

    # Create a mask for the inspiratory phase (u_out == 0)
    # Using < 0.5 to handle potential float types safely, though u_out is binary
    mask = u_out < 0.5

    # If no inspiratory phase data is present (unlikely), return 0.0
    if mask.sum() == 0:
        return 0.0

    # Filter predictions and targets
    y_pred_insp = y_pred[mask]
    y_true_insp = y_true[mask]

    # Calculate Mean Absolute Error
    mae = torch.abs(y_pred_insp - y_true_insp).mean()

    return mae.item()
