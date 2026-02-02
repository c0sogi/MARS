import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=1.0, device=None):
    """
    Applies Mixup augmentation to the input batch.

    Args:
        x (torch.Tensor): Input batch.
        y (torch.Tensor): Target labels.
        alpha (float): Mixup alpha parameter.
        device (torch.device, optional): Device to store the shuffle index.
                                         If None, uses x.device.

    Returns:
        mixed_x (torch.Tensor): Mixed input batch.
        y_a (torch.Tensor): Targets for the first component.
        y_b (torch.Tensor): Targets for the second component.
        lam (float): Lambda value used for mixing.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if device is None:
        device = x.device

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the Mixup loss.

    Args:
        criterion (callable): Loss function.
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Targets for the first component.
        y_b (torch.Tensor): Targets for the second component.
        lam (float): Lambda value.

    Returns:
        loss (torch.Tensor): Weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def save_checkpoint(state, is_best, filename="checkpoint.pth", save_dir=None):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model parameters, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Filename for the checkpoint.
        save_dir (str, optional): Directory to save the checkpoint. Defaults to Config.WORKING_DIR.
    """
    if save_dir is None:
        save_dir = Config.WORKING_DIR

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(save_dir, "best_model.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(
    model, optimizer=None, filename="best_model.pth", load_dir=None, device=None
):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        filename (str): Filename of the checkpoint.
        load_dir (str, optional): Directory where the checkpoint is located. Defaults to Config.WORKING_DIR.
        device (str or torch.device, optional): Device to map the location to. Defaults to Config.DEVICE.

    Returns:
        checkpoint (dict): The loaded checkpoint dictionary.
        start_epoch (int): The epoch to resume from.
        best_score (float): The best score recorded in the checkpoint.
    """
    if load_dir is None:
        load_dir = Config.WORKING_DIR

    filepath = os.path.join(load_dir, filename)

    if not os.path.isfile(filepath):
        print(f"No checkpoint found at '{filepath}'")
        return None, 0, 0.0

    if device is None:
        device = Config.DEVICE

    print(f"Loading checkpoint '{filepath}'...")
    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Load optimizer state
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", 0.0)

    print(f"Loaded checkpoint '{filename}' (epoch {start_epoch}, score {best_score})")

    return checkpoint, start_epoch, best_score
