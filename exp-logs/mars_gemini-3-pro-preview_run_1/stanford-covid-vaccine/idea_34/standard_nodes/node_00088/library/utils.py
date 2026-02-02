import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.seed):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_loss(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This function computes the RMSE for each target column separately and then
    returns the average of these RMSEs. It expects inputs to be tensors.

    Args:
        y_true (torch.Tensor): Ground truth tensor. Shape: (Batch, Seq_Len, Targets) or (N, Targets).
        y_pred (torch.Tensor): Predicted tensor. Shape: (Batch, Seq_Len, Targets) or (N, Targets).

    Returns:
        torch.Tensor: The scalar MCRMSE loss.
    """
    # Flatten the batch and sequence dimensions to (N, num_targets)
    # This ensures we calculate the RMSE over all predictions for a specific target column
    y_true_flat = y_true.view(-1, y_true.shape[-1])
    y_pred_flat = y_pred.view(-1, y_pred.shape[-1])

    # Calculate MSE for each column
    colwise_mse = torch.mean((y_true_flat - y_pred_flat) ** 2, dim=0)

    # Calculate RMSE for each column
    colwise_rmse = torch.sqrt(colwise_mse)

    # Return the mean of the column RMSEs
    return torch.mean(colwise_rmse)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves the model and optimizer state to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        loss (float): The validation loss at this checkpoint.
        path (str): The file path to save the checkpoint.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": loss,
    }
    torch.save(state, path)


def load_checkpoint(model, optimizer, path, device=Config.device):
    """
    Loads the model and optimizer state from a file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        path (str): The file path to load the checkpoint from.
        device (str): The device to map the checkpoint to.

    Returns:
        tuple: (epoch, loss) if successful, otherwise None.
    """
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        if checkpoint["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))

    return epoch, loss
