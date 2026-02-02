import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=Config.MIXUP_ALPHA, device=Config.DEVICE):
    """
    Applies Mixup augmentation to the input batch.

    Args:
        x (torch.Tensor): Input batch of data (e.g., spectrograms).
        y (torch.Tensor): Target labels for the batch.
        alpha (float): Parameter for the Beta distribution. Defaults to Config.MIXUP_ALPHA.
        device (torch.device): The device to place tensors on. Defaults to Config.DEVICE.

    Returns:
        mixed_x (torch.Tensor): The mixed input batch.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Shuffled targets.
        lam (float): The mixing coefficient (lambda).
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the loss for Mixup training.

    Args:
        criterion (callable): The loss function (e.g., CrossEntropyLoss).
        pred (torch.Tensor): The model predictions.
        y_a (torch.Tensor): Original targets.
        y_b (torch.Tensor): Shuffled targets.
        lam (float): The mixing coefficient.

    Returns:
        loss (torch.Tensor): The weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def save_checkpoint(
    state, is_best, checkpoint_dir=Config.WORKING_DIR, filename="checkpoint.pth"
):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the checkpoint. Defaults to Config.WORKING_DIR.
        filename (str): Name of the checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, device=Config.DEVICE
):
    """
    Loads a model checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        device (torch.device): Device to map the storage to.

    Returns:
        epoch (int): The epoch at which the checkpoint was saved.
        best_acc (float): The best accuracy recorded in the checkpoint.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    # Handle case where DataParallel was used (keys start with 'module.')
    state_dict = checkpoint["state_dict"]
    if list(state_dict.keys())[0].startswith("module."):
        new_state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    # Load optimizer state
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = checkpoint.get("epoch", 0)
    best_acc = checkpoint.get("best_acc", 0.0)

    return epoch, best_acc
