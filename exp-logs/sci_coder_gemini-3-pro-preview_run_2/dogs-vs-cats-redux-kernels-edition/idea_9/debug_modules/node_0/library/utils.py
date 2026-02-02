import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Seeds all random number generators for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and accuracy during training.
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


def mixup_data(x, y, alpha=Config.MIXUP_ALPHA, device=Config.DEVICE):
    """
    Applies Mixup augmentation to the batch.

    Args:
        x (torch.Tensor): Input images.
        y (torch.Tensor): Target labels.
        alpha (float): Mixup alpha parameter.
        device (str): Device to perform operations on.

    Returns:
        mixed_x (torch.Tensor): Mixed input images.
        y_a (torch.Tensor): Targets for the first image.
        y_b (torch.Tensor): Targets for the second image.
        lam (float): Lambda value used for mixing.
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
    Calculates the loss for Mixup.

    Args:
        criterion: Loss function (e.g., CrossEntropyLoss).
        pred: Model predictions.
        y_a: First set of targets.
        y_b: Second set of targets.
        lam: Mixing coefficient.

    Returns:
        loss: Weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def save_checkpoint(state, is_best, filename, dir_path=Config.CHECKPOINT_DIR):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model, optimizer, etc.
        is_best (bool): Whether this checkpoint represents the best metric so far.
        filename (str): Name of the checkpoint file.
        dir_path (str): Directory to save the checkpoint.
    """
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, filename)
    torch.save(state, filepath)

    if is_best:
        # Create a specific best filename to avoid overwriting other folds/models
        # e.g., if filename is 'convnext_fold0.pth', best is 'best_convnext_fold0.pth'
        best_filename = f"best_{filename}"
        best_filepath = os.path.join(dir_path, best_filename)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device=Config.DEVICE):
    """
    Loads a model checkpoint.

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): Model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.
        device (str): Device to map the location to.

    Returns:
        checkpoint (dict): The loaded checkpoint dictionary.
    """
    if not os.path.exists(path):
        print(f"Checkpoint not found at {path}")
        return None

    checkpoint = torch.load(path, map_location=device)

    # Handle both 'state_dict' key and direct state dict
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Remove 'module.' prefix if present (from DataParallel)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint
