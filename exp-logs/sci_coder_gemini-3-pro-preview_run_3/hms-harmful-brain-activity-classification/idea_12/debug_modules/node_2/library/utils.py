import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from library.config import Config, seed_everything


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def kl_divergence(y_true, y_pred, epsilon=1e-15):
    """
    Calculates the Kullback-Leibler Divergence between the predicted probability
    and the observed target.

    Args:
        y_true (np.array): Ground truth probabilities. Shape (N, C).
        y_pred (np.array): Predicted probabilities. Shape (N, C).
        epsilon (float): Small value to prevent log(0).

    Returns:
        float: The average KL divergence.
    """
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Calculate KL Divergence: sum(y_true * log(y_true / y_pred))
    # Note: y_true * log(y_true) is 0 where y_true is 0
    # We can expand: y_true * (log(y_true) - log(y_pred))

    # Handle the y_true = 0 case safely
    # We use a mask where y_true > 0 to compute the log(y_true) part
    # Or simply: sum(p * log(p/q))

    # Element-wise calculation
    terms = y_true * np.log(y_true / y_pred)

    # Where y_true is 0, the term should be 0 (limit x->0 of x*log(x) is 0)
    terms[y_true == 0] = 0.0

    # Sum over classes (axis=1), then mean over samples (axis=0)
    return np.mean(np.sum(terms, axis=1))


def save_checkpoint(
    state, is_best, filename="checkpoint.pth", best_filename="best_model.pth"
):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint is the best so far.
        filename (str): Filename for the current checkpoint.
        best_filename (str): Filename for the best model.
    """
    # Ensure directory exists
    save_dir = Config.WORKING_DIR
    filepath = os.path.join(save_dir, filename)

    torch.save(state, filepath)

    if is_best:
        best_filepath = os.path.join(save_dir, best_filename)
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(model, filename, device=None, optimizer=None, scheduler=None):
    """
    Loads a model checkpoint.

    Args:
        model (torch.nn.Module): The model to load weights into.
        filename (str): Path to the checkpoint file.
        device (str, optional): Device to load the model onto.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (optional): Scheduler to load state into.

    Returns:
        int: The epoch to resume from (start_epoch).
        float: The best metric value (best_score).
    """
    if not os.path.exists(filename):
        print(f"No checkpoint found at '{filename}'")
        return 0, float("inf")

    if device is None:
        device = Config.DEVICE

    checkpoint = torch.load(filename, map_location=device, weights_only=False)

    # Load model weights
    # Handle DataParallel wrapping if necessary (remove 'module.' prefix)
    state_dict = checkpoint["state_dict"]
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    # Load optimizer and scheduler if provided
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", float("inf"))

    print(f"Loaded checkpoint '{filename}' (epoch {epoch}, best_score {best_score})")
    return epoch, best_score
