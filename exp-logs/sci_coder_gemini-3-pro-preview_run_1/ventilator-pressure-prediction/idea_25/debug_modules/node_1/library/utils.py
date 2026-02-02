import os
import random
import time
import numpy as np
import torch
from functools import wraps
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the torch device based on availability and config.

    Returns:
        torch.device: The device object (cpu or cuda).
    """
    return torch.device(Config.DEVICE)


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


def time_execution(func):
    """
    Decorator to measure and print the execution time of a function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start} seconds")
        return result

    return wrapper


def compute_mae(preds, targets, u_out=None):
    """
    Computes the Mean Absolute Error (MAE).

    Args:
        preds (torch.Tensor or np.ndarray): Predicted values.
        targets (torch.Tensor or np.ndarray): Ground truth values.
        u_out (torch.Tensor or np.ndarray, optional): Control input for expiratory valve.
                                                      If provided, calculates MAE only for
                                                      inspiratory phase (u_out == 0).

    Returns:
        float: The calculated MAE.
    """
    # Convert to torch tensors if numpy arrays
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if u_out is not None and isinstance(u_out, np.ndarray):
        u_out = torch.from_numpy(u_out)

    # Ensure flat tensors
    preds = preds.view(-1)
    targets = targets.view(-1)
    if u_out is not None:
        u_out = u_out.view(-1)

    # Move to CPU for calculation to avoid GPU sync overhead if just for metrics
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    if u_out is not None:
        u_out = u_out.detach().cpu()
        # Filter for inspiratory phase (u_out == 0)
        mask = u_out == 0
        # Apply mask
        preds = preds[mask]
        targets = targets[mask]

    return torch.mean(torch.abs(preds - targets)).item()


def save_checkpoint(state, is_best=False, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        is_best (bool): If True, saves a copy as the best model.
        filename (str): Filename for the checkpoint.
    """
    save_dir = Config.WORKING_DIR
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(save_dir, "model_best.pth")
        torch.save(state, best_path)


def load_checkpoint(model, optimizer=None, scheduler=None, filename="model.pth"):
    """
    Loads a checkpoint into the model, optimizer, and scheduler.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        filename (str): The filename of the checkpoint to load.

    Returns:
        tuple: (start_epoch, best_loss)
               start_epoch (int): The epoch to resume from (if found in checkpoint), else 0.
               best_loss (float): The best loss (if found), else inf.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    if not os.path.exists(filepath):
        # Fallback to Config.MODEL_PATH if generic filename not found
        if os.path.exists(Config.MODEL_PATH):
            filepath = Config.MODEL_PATH
        else:
            print(f"No checkpoint found at {filepath}")
            return 0, float("inf")

    print(f"Loading checkpoint from {filepath}")
    checkpoint = torch.load(filepath, map_location=Config.DEVICE)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0)
    best_loss = checkpoint.get("best_loss", float("inf"))

    return start_epoch, best_loss
