import os
import random
import numpy as np
import torch
from copy import deepcopy
from sklearn.metrics import f1_score


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModelEMA:
    """
    Implements Exponential Moving Average (EMA) of model parameters.
    This helps in stabilizing the training and often yields better generalization,
    especially in the presence of noisy labels.
    """

    def __init__(self, model, decay=0.9998, device=None):
        """
        Initialize the EMA model.

        Args:
            model (nn.Module): The model to track.
            decay (float): The decay factor for EMA (default: 0.9998).
            device (torch.device): Device to store the EMA model. If None, uses model's device.
        """
        # Create a deep copy of the model for EMA
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device if device else next(model.parameters()).device
        self.module.to(self.device)

        # Ensure EMA parameters do not require gradients
        for param in self.module.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the EMA model parameters using the current model parameters.

        Args:
            model (nn.Module): The current training model.
        """
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.module.state_dict()

            for k in msd.keys():
                model_v = msd[k].detach()
                ema_v = esd[k]

                if model_v.device != ema_v.device:
                    model_v = model_v.to(ema_v.device)

                # Only apply EMA to floating point parameters (weights, biases, running stats)
                # Integer buffers (like num_batches_tracked) should be copied directly
                if ema_v.is_floating_point():
                    ema_v.copy_(ema_v * self.decay + model_v * (1.0 - self.decay))
                else:
                    ema_v.copy_(model_v)


def calculate_f1(preds, targets):
    """
    Calculates the Micro F1 score.

    Args:
        preds (np.array or torch.Tensor): Binary predictions.
        targets (np.array or torch.Tensor): Ground truth labels.

    Returns:
        float: Micro F1 score.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    return f1_score(targets, preds, average="micro")


def optimize_threshold(val_probs, val_targets):
    """
    Finds the optimal single global threshold that maximizes the Micro F1 score.
    Strictly avoids per-class thresholding.

    Args:
        val_probs (np.array or torch.Tensor): Predicted probabilities (N, C).
        val_targets (np.array or torch.Tensor): Ground truth binary labels (N, C).

    Returns:
        tuple: (best_threshold, best_f1_score)
    """
    if isinstance(val_probs, torch.Tensor):
        val_probs = val_probs.detach().cpu().numpy()
    if isinstance(val_targets, torch.Tensor):
        val_targets = val_targets.detach().cpu().numpy()

    best_threshold = 0.5
    best_score = 0.0

    # Iterate over a range of thresholds to find the optimum
    # Using a step of 0.01 provides sufficient granularity
    thresholds = np.arange(0.01, 1.00, 0.01)

    for thresh in thresholds:
        # Apply threshold
        preds = (val_probs >= thresh).astype(int)

        # Calculate score
        score = f1_score(val_targets, preds, average="micro")

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score
