import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_rmse(predictions, targets):
    """
    Computes the Root Mean Squared Error (RMSE) between predictions and targets.
    Handles both PyTorch tensors and NumPy arrays.

    Args:
        predictions (torch.Tensor or np.ndarray): The predicted pixel intensities.
        targets (torch.Tensor or np.ndarray): The ground truth pixel intensities.

    Returns:
        float: The calculated RMSE value.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs are float32 for precision and flatten if necessary for element-wise op consistency
    # (Though numpy handles shapes well, flattening ensures we strictly compute over all pixels)
    predictions = predictions.astype(np.float32)
    targets = targets.astype(np.float32)

    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def save_checkpoint(
    model, optimizer, scheduler, epoch, loss, filename=Config.MODEL_SAVE_PATH
):
    """
    Saves the model checkpoint including optimizer and scheduler states.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler state (can be None).
        epoch (int): The current epoch number.
        loss (float): The validation loss at this checkpoint.
        filename (str): The path to save the checkpoint.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, filename)


def load_checkpoint(
    model,
    optimizer=None,
    scheduler=None,
    filename=Config.MODEL_SAVE_PATH,
    device=Config.DEVICE,
):
    """
    Loads a model checkpoint. Returns the epoch and loss from the checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        filename (str): The path to the checkpoint file.
        device (str): The device to map the checkpoint to.

    Returns:
        tuple: (epoch, loss) loaded from the checkpoint. Returns (0, inf) if file not found.
    """
    if not os.path.exists(filename):
        # This case might happen if starting training from scratch
        return 0, float("inf")

    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))

    return epoch, loss
