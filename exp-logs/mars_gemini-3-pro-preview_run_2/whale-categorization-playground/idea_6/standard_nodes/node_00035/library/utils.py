import os
import random
import shutil
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Ensures deterministic behavior for CUDA operations.

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
        # Enforce deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and accuracy during training epochs.
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


def map_5(preds, targets):
    """
    Calculates Mean Average Precision @ 5 (MAP@5).

    The metric corresponds to the average of 1/rank of the correct label,
    provided the correct label is within the top 5 predictions.

    Args:
        preds (list or np.ndarray): Predicted labels. Shape (N, 5) or list of lists.
                                    Each row contains the top 5 predicted class IDs.
        targets (list or np.ndarray): Ground truth labels. Shape (N,).
                                      Each entry is the correct class ID.

    Returns:
        float: The MAP@5 score.
    """
    n = len(targets)
    if n == 0:
        return 0.0

    score = 0.0
    for p, t in zip(preds, targets):
        # Convert prediction to list to safely find index
        p_list = list(p)

        # Ensure we only consider the top 5
        if len(p_list) > 5:
            p_list = p_list[:5]

        if t in p_list:
            # Rank is 1-based index
            rank = p_list.index(t) + 1
            score += 1.0 / rank

    return score / n


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model parameters, optimizer, epoch, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Filename to save the current checkpoint.
    """
    # Ensure the directory exists
    save_dir = os.path.dirname(filename)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    torch.save(state, filename)

    if is_best:
        best_path = Config.MODEL_PATH
        # Ensure directory for best model exists
        best_dir = os.path.dirname(best_path)
        if best_dir:
            os.makedirs(best_dir, exist_ok=True)
        shutil.copyfile(filename, best_path)


def load_checkpoint(model, optimizer=None, scheduler=None, filename=None):
    """
    Loads a checkpoint into the model, optimizer, and scheduler.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        filename (str, optional): Path to the checkpoint file. Defaults to Config.MODEL_PATH.

    Returns:
        tuple: (start_epoch, best_score)
            start_epoch (int): The epoch to resume training from.
            best_score (float): The best validation score recorded in the checkpoint.
    """
    if filename is None:
        filename = Config.MODEL_PATH

    if not os.path.exists(filename):
        return 0, 0.0

    # Load on the configured device
    checkpoint = torch.load(filename, map_location=Config.DEVICE)

    # Load model state
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Fallback for simple weight saves
        model.load_state_dict(checkpoint)

    # Load optimizer state
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_score = checkpoint.get("best_score", 0.0)

    return start_epoch, best_score
