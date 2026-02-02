import os
import torch
import numpy as np
import random
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across all libraries.
    Delegates to the centralized Config configuration.
    """
    Config.set_seed(seed)


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """
    Saves the model checkpoint including optimizer and scheduler states.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler state.
        epoch (int): The current epoch number.
        path (str): The file path to save the checkpoint.
    """
    # Ensure directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
    }

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device=Config.DEVICE):
    """
    Loads a model checkpoint.

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into.
        device (str): Device to map the storage to.

    Returns:
        int: The epoch number associated with the checkpoint (returns 0 if not found).
    """
    if not os.path.exists(path):
        # Return 0 to indicate starting from scratch if no checkpoint found
        return 0

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0)


def calculate_accuracy(outputs, targets, ignore_index=None):
    """
    Calculates the accuracy of predictions against targets.

    Args:
        outputs (torch.Tensor): Predictions from the model.
                                Can be logits [Batch, ..., Classes] or indices [Batch, ...].
        targets (torch.Tensor): Ground truth indices [Batch, ...].
        ignore_index (int, optional): Index to ignore in calculation (e.g., padding).

    Returns:
        float: The accuracy (correct / total).
    """
    with torch.no_grad():
        # If outputs are logits (dimension higher than targets), take argmax
        if outputs.dim() > targets.dim():
            # Assume logits are in the last dimension [B, Seq, Class]
            # or second dimension [B, Class, Seq].
            # Standard CrossEntropy expects [B, Class, ...], but standard output is often [B, ..., Class]
            # We infer based on shape matching.
            if (
                outputs.size(-1) != targets.size(-1)
                and outputs.dim() == targets.dim() + 1
            ):
                # Case [B, L, C] vs [B, L] -> argmax over C (last dim)
                outputs = torch.argmax(outputs, dim=-1)
            elif (
                outputs.size(1) != targets.size(1)
                and outputs.dim() == targets.dim() + 1
            ):
                # Case [B, C, L] vs [B, L] -> argmax over C (dim 1)
                outputs = torch.argmax(outputs, dim=1)
            else:
                # Default fallback to last dimension
                outputs = torch.argmax(outputs, dim=-1)

        # Flatten tensors to 1D
        outputs = outputs.reshape(-1)
        targets = targets.reshape(-1)

        # Apply mask if ignore_index is provided
        if ignore_index is not None:
            mask = targets != ignore_index
            outputs = outputs[mask]
            targets = targets[mask]

        if targets.numel() == 0:
            return 0.0

        correct = (outputs == targets).sum().item()
        total = targets.numel()

        return correct / total
