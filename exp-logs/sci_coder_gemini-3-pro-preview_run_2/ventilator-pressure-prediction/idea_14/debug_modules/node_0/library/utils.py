import os
import random
import shutil
import numpy as np
import torch
from collections import defaultdict


def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricMonitor:
    """
    A utility class to track and compute running averages of metrics (e.g., Loss, MAE).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the internal metric storage."""
        self.val = defaultdict(float)
        self.sum = defaultdict(float)
        self.count = defaultdict(int)
        self.avg = defaultdict(float)

    def update(self, metric_name, val, n=1):
        """
        Updates the metric statistics.

        Args:
            metric_name (str): The name of the metric.
            val (float): The current value of the metric (e.g., batch mean).
            n (int): The number of samples associated with the value (e.g., batch size).
        """
        self.val[metric_name] = val
        self.sum[metric_name] += val * n
        self.count[metric_name] += n
        self.avg[metric_name] = self.sum[metric_name] / self.count[metric_name]

    def __str__(self):
        """
        Returns a string representation of the metrics with full precision.
        """
        return " | ".join(
            [
                "{}: {}".format(metric_name, self.avg[metric_name])
                for metric_name in self.avg
            ]
        )


def save_checkpoint(state, is_best, checkpoint_dir, best_model_name="best_model.pth"):
    """
    Saves the model state to a checkpoint file.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether the current model is the best one found so far.
        checkpoint_dir (str): The directory where checkpoints should be saved.
        best_model_name (str): The filename to use for the best model copy.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, "checkpoint.pth")
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, best_model_name)
        shutil.copyfile(filepath, best_path)


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, device="cpu"
):
    """
    Loads a checkpoint into the model, optimizer, and scheduler.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model instance to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (optional): The scheduler to load state into.
        device (str): The device to map the checkpoint location to.

    Returns:
        start_epoch (int): The epoch to resume training from.
        best_score (float): The best score recorded in the checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state dict, stripping 'module.' prefix if model was saved using DataParallel
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    # Load optimizer state
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Load scheduler state
    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_score = checkpoint.get("best_score", float("inf"))

    return start_epoch, best_score
