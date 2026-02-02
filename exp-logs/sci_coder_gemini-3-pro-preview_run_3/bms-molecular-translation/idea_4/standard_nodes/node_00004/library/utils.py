import os
import shutil
import random
import numpy as np
import torch
from nltk.metrics import distance
from library.config import Config


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and metrics during training.
    """

    def __init__(self, name=None):
        self.name = name
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


def compute_levenshtein(predictions, targets):
    """
    Computes the mean Levenshtein distance between predicted and target strings.

    Args:
        predictions (list[str]): List of predicted InChI strings.
        targets (list[str]): List of ground truth InChI strings.

    Returns:
        float: The mean Levenshtein distance.
    """
    if not predictions or not targets:
        return 0.0

    total_dist = 0
    n = len(predictions)

    for pred, target in zip(predictions, targets):
        # nltk.metrics.distance.edit_distance computes Levenshtein distance
        dist = distance.edit_distance(pred, target)
        total_dist += dist

    return total_dist / n


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the training checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Boolean flag indicating if this is the best model so far.
        filename (str): Filename for the checkpoint.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    filepath = os.path.join(Config.WORKING_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        # Save to the specific best model path defined in Config
        shutil.copyfile(filepath, Config.MODEL_PATH)


def load_checkpoint(
    model, optimizer=None, scheduler=None, path=None, device=Config.DEVICE
):
    """
    Loads a checkpoint into the model and optionally optimizer/scheduler.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.
        path (str, optional): Path to the checkpoint file. Defaults to Config.MODEL_PATH.
        device (torch.device): Device to map the checkpoint to.

    Returns:
        tuple: (start_epoch, best_metric)
    """
    if path is None:
        path = Config.MODEL_PATH

    if not os.path.exists(path):
        print(f"Checkpoint not found at '{path}'. Starting from scratch.")
        return 0, float("inf")

    print(f"Loading checkpoint from '{path}'...")
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_metric = checkpoint.get("best_metric", float("inf"))

    print(f"Loaded checkpoint (epoch {start_epoch-1}, best_metric {best_metric})")
    return start_epoch, best_metric


def seed_everything(seed=Config.SEED):
    """
    Seeds all random number generators for reproducibility.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Counts the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): The model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
