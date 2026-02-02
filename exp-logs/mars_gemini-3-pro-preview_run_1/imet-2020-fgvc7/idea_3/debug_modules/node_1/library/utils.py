import os
import torch
import copy
import numpy as np
from sklearn.metrics import f1_score
from library.config import Config, seed_everything


def calculate_f1(preds, targets):
    """
    Calculates the Micro-F1 score.

    Args:
        preds (np.array): Binary predictions (0 or 1), shape (N, num_classes).
        targets (np.array): Ground truth labels (0 or 1), shape (N, num_classes).

    Returns:
        float: The micro-averaged F1 score.
    """
    return f1_score(targets, preds, average="micro")


class ModelEMA:
    """
    Implements Exponential Moving Average (EMA) for model parameters.
    Maintains a shadow model that updates slowly based on the training model.
    """

    def __init__(self, model, decay=None):
        """
        Args:
            model (torch.nn.Module): The model to track.
            decay (float, optional): The decay factor. Defaults to Config.EMA_DECAY.
        """
        self.decay = decay if decay is not None else Config.EMA_DECAY
        self.ema = copy.deepcopy(model)
        self.ema.eval()

        # Disable gradients for the shadow model
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model):
        """
        Update the shadow model parameters using the current model parameters.

        Args:
            model (torch.nn.Module): The current training model.
        """
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()

            for k, v in esd.items():
                model_v = msd[k].detach()

                if v.dtype.is_floating_point:
                    # Apply EMA: shadow = decay * shadow + (1 - decay) * new
                    v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
                else:
                    # Directly copy non-floating point parameters (e.g., num_batches_tracked)
                    v.copy_(model_v)


def save_checkpoint(
    model, optimizer, scheduler, epoch, score, filename="checkpoint.pth"
):
    """
    Saves the training checkpoint.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        epoch (int): Current epoch.
        score (float): Validation score (Micro F1).
        filename (str): Name of the file to save in Config.WORKING_DIR.
    """
    filepath = os.path.join(Config.WORKING_DIR, filename)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "score": score,
    }
    torch.save(state, filepath)


def load_checkpoint(
    filepath, model, optimizer=None, scheduler=None, device=Config.DEVICE
):
    """
    Loads a checkpoint.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The scheduler to load state into.
        device (str): Device to map the location to.

    Returns:
        dict: The checkpoint dictionary containing epoch and score.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")

    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"]
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
