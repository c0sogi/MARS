import os
import random
import shutil
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
    torch.cuda.manual_seed_all(seed)
    # Deterministic operations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metric(preds, targets, u_out):
    """
    Computes the Mean Absolute Error (MAE) for the inspiratory phase.
    The competition metric only scores time steps where u_out == 0.

    Args:
        preds (torch.Tensor or np.ndarray): Predicted pressure values.
        targets (torch.Tensor or np.ndarray): Ground truth pressure values.
        u_out (torch.Tensor or np.ndarray): Control input indicating phase (0=Inspiratory, 1=Expiratory).

    Returns:
        float: The MAE for the inspiratory phase.
    """
    # Convert to tensor if numpy array
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if isinstance(u_out, np.ndarray):
        u_out = torch.from_numpy(u_out)

    # Flatten inputs
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    u_out = u_out.reshape(-1)

    # Create mask for inspiratory phase (u_out == 0)
    mask = u_out == 0

    # Avoid division by zero if mask is empty (though unlikely in valid batches)
    if mask.sum() == 0:
        return 0.0

    # Calculate MAE on masked data
    mae = torch.abs(preds[mask] - targets[mask]).mean()

    return mae.item()


def save_checkpoint(state, is_best, checkpoint_dir):
    """
    Saves the model checkpoint to the specified directory.

    Args:
        state (dict): State dictionary containing model params, optimizer, epoch, etc.
        is_best (bool): If True, copies the checkpoint to 'best_model.pth'.
        checkpoint_dir (str): Directory to save the checkpoint files.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, "checkpoint.pth")
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"
):
    """
    Loads a checkpoint into the model and optionally optimizer and scheduler.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (optional): Learning rate scheduler to load state into.
        device (str): Device to map the checkpoint to.

    Returns:
        dict: The full checkpoint dictionary (useful for retrieving epoch or best score).
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)  # Handle raw state dict saves

    # Load optimizer state if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state if provided
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint
