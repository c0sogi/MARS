import os
import shutil
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
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


def save_checkpoint(
    state, is_best, filename="checkpoint.pth", best_filename="best_model.pth"
):
    """
    Saves the training checkpoint.

    Args:
        state (dict): The state dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filename (str): Name of the checkpoint file.
        best_filename (str): Name of the best model file.
    """
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(Config.CHECKPOINT_DIR, best_filename)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(model, filename, optimizer=None, scheduler=None, device=None):
    """
    Loads a checkpoint into the model and optionally the optimizer and scheduler.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filename (str): The filename of the checkpoint to load.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        device (torch.device, optional): The device to map the storage to.

    Returns:
        dict: The loaded checkpoint dictionary (useful for retrieving epoch or best_score).
    """
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No checkpoint found at '{filepath}'")

    # Map to the correct device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint


def get_scheduler(optimizer, mode, **kwargs):
    """
    Factory function to get the appropriate learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer.
        mode (str): 'plateau' for ReduceLROnPlateau (Phase 1) or 'step' for MultiStepLR (Phase 3).
        **kwargs: Additional arguments for the scheduler.
            - For 'step', 'milestones' (list of ints) is required.

    Returns:
        scheduler: The initialized PyTorch scheduler.
    """
    if mode == "plateau":
        # Phase 1: Reactive scheduling based on validation metric
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
        )
    elif mode == "step":
        # Phase 3: Trajectory Replay using fixed milestones
        if "milestones" not in kwargs:
            raise ValueError("Scheduler mode 'step' requires 'milestones' in kwargs.")

        milestones = kwargs["milestones"]
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=Config.SCHEDULER_FACTOR
        )
    else:
        raise ValueError(f"Unknown scheduler mode: {mode}")
