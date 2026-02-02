import os
import random
import numpy as np
import torch
import shutil
from copy import deepcopy
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the Config class to ensure consistency.
    """
    Config.set_seed(seed)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking loss and accuracy during training epochs.
    """

    def __init__(self, name="Metric", fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def calculate_accuracy(output, target):
    """
    Computes the accuracy given the model output (logits) and targets.

    Args:
        output (torch.Tensor): Logits of shape (batch_size, num_classes)
        target (torch.Tensor): Ground truth labels of shape (batch_size)

    Returns:
        float: Accuracy percentage (0-100)
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        res = correct_k.mul_(100.0 / batch_size)
        return res.item()


class ModelEMA:
    """
    Implements Exponential Moving Average (EMA) of model parameters.
    Maintains a shadow copy of the model that is updated as a moving average
    of the training model weights. This often leads to better generalization.
    """

    def __init__(self, model, decay=0.999, device=None):
        """
        Args:
            model (nn.Module): The model to track.
            decay (float): The decay factor for the moving average.
            device (torch.device): Device to store the shadow model on.
        """
        self.decay = decay
        # Create a deep copy of the model for the shadow weights
        self.ema = deepcopy(model)
        self.ema.eval()

        if device is not None:
            self.ema.to(device)

        # Ensure EMA parameters do not require gradients
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the shadow parameters based on the current model parameters.

        Args:
            model (nn.Module): The current training model.
        """
        with torch.no_grad():
            # Iterate over state_dict to handle both parameters and buffers (e.g., BatchNorm stats)
            msd = model.state_dict()
            esd = self.ema.state_dict()

            for name, param in msd.items():
                if name in esd:
                    if param.dtype.is_floating_point:
                        # EMA update for floating point parameters/buffers
                        esd[name].copy_(
                            esd[name] * self.decay + param * (1.0 - self.decay)
                        )
                    else:
                        # Direct copy for integer buffers (e.g., num_batches_tracked)
                        esd[name].copy_(param)

    def get_model(self):
        """Returns the shadow model."""
        return self.ema


def save_checkpoint(state, is_best, filepath, filename="checkpoint.pth"):
    """
    Saves the training checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        filepath (str): Path to the file where the best model should be saved.
        filename (str): Name for the generic checkpoint file (optional).
    """
    # Save best model directly to the target filepath
    if is_best:
        torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device="cpu"):
    """
    Loads a checkpoint into the model and optionally optimizer/scheduler.

    Args:
        filepath (str): Path to the checkpoint file.
        model (nn.Module): Model to load weights into.
        optimizer (Optimizer, optional): Optimizer to load state into.
        scheduler (Scheduler, optional): Scheduler to load state into.
        device (str): Device to map the storage to.

    Returns:
        int: The epoch to resume from (0 if failed/not found).
        float: The best metric score (0.0 if failed/not found).
    """
    if not os.path.exists(filepath):
        return 0, 0.0

    checkpoint = torch.load(filepath, map_location=device)

    # Load model weights
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Fallback if just the state dict was saved
        model.load_state_dict(checkpoint)

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", 0.0)

    return start_epoch, best_score
