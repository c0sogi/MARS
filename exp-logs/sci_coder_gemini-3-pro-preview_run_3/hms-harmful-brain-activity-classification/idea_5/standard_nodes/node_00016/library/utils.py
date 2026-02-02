import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed: int):
    """
    Sets the random seed for various libraries to ensure reproducibility.
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
    Used for tracking loss and metrics during training.
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


class KLDivLossWithLogits(nn.Module):
    """
    Kullback-Leibler Divergence Loss wrapper.

    This module expects the input `y_pred` to be raw logits (unbounded)
    and `y_true` to be probability distributions (summing to 1).

    It applies LogSoftmax to `y_pred` before computing the KL Divergence,
    ensuring numerical stability and compatibility with PyTorch's nn.KLDivLoss.
    """

    def __init__(self, reduction="batchmean"):
        super(KLDivLossWithLogits, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): Predicted logits of shape (batch_size, num_classes).
            y_true (torch.Tensor): Target probabilities of shape (batch_size, num_classes).

        Returns:
            torch.Tensor: The calculated loss.
        """
        # Apply log_softmax to logits to get log-probabilities
        log_probs = F.log_softmax(y_pred, dim=1)

        # Calculate KL Divergence
        # nn.KLDivLoss expects input=log_probs, target=probs
        loss = F.kl_div(log_probs, y_true, reduction=self.reduction)
        return loss


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """
    Saves the model, optimizer, and scheduler states to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        epoch (int): Current epoch.
        loss (float): Current validation loss.
        path (str): File path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, path)


def load_checkpoint(model, path, device, optimizer=None, scheduler=None):
    """
    Loads a checkpoint into the model, optimizer, and scheduler.

    Args:
        model (torch.nn.Module): The model to load weights into.
        path (str): Path to the checkpoint file.
        device (str): Device to map the location to ('cpu' or 'cuda').
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.

    Returns:
        tuple: (epoch, loss) from the checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))

    return epoch, loss
