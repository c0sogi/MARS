import os
import random
import copy
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=42):
    """
    Seeds all random number generators for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Returns the device specified in Config or detects available device.

    Returns:
        torch.device: The computing device.
    """
    return torch.device(Config.DEVICE)


class ModelEma:
    """
    Exponential Moving Average (EMA) for model weights.
    Maintains a shadow copy of the model that is updated using EMA.

    Reference: https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/model_ema.py
    """

    def __init__(self, model, decay=0.999, device=None):
        """
        Args:
            model (torch.nn.Module): The model to track.
            decay (float): The decay factor for EMA (default: 0.999).
            device (torch.device, optional): Device to store the shadow model on.
        """
        self.decay = decay
        # Create a deep copy of the model to serve as the shadow model
        self.model = copy.deepcopy(model)
        self.model.eval()

        # Determine device
        if device is None:
            device = get_device()
        self.model.to(device)

        # Freeze parameters in the shadow model so they aren't updated by the optimizer
        for param in self.model.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the shadow model parameters using the current model parameters.

        Args:
            model (torch.nn.Module): The current training model.
        """
        # Handle DataParallel/DistributedDataParallel wrappers
        if hasattr(model, "module"):
            model = model.module

        with torch.no_grad():
            # Update parameters
            model_params = dict(model.named_parameters())
            shadow_params = dict(self.model.named_parameters())

            for name, param in shadow_params.items():
                if name in model_params:
                    # EMA update formula: shadow = decay * shadow + (1 - decay) * current
                    new_param = model_params[name].data.to(param.device)
                    param.data.mul_(self.decay).add_(new_param, alpha=1 - self.decay)

            # Copy buffers (e.g., BatchNorm running mean/var) directly
            model_buffers = dict(model.named_buffers())
            shadow_buffers = dict(self.model.named_buffers())

            for name, buffer in shadow_buffers.items():
                if name in model_buffers:
                    new_buffer = model_buffers[name].data.to(buffer.device)
                    buffer.data.copy_(new_buffer)

    def apply_shadow(self):
        """
        Returns the shadow model (the EMA model) for inference.

        Returns:
            torch.nn.Module: The EMA model.
        """
        return self.model


def calculate_metric(y_true, y_pred):
    """
    Calculates the Mean Column-wise ROC AUC.

    Args:
        y_true (np.ndarray): Ground truth labels (one-hot or binary), shape (N, num_classes).
        y_pred (np.ndarray): Predicted probabilities, shape (N, num_classes).

    Returns:
        float: Mean ROC AUC score.
    """
    try:
        # average='macro' calculates the metric for each label independently and returns the unweighted mean.
        score = roc_auc_score(y_true, y_pred, average="macro")
        return score
    except ValueError:
        # This can happen if a batch contains only one class for a specific target.
        # In a full validation set, this shouldn't occur given the stratification.
        return 0.5


def save_checkpoint(state, filepath):
    """
    Saves the model checkpoint to the specified filepath.

    Args:
        state (dict): State dictionary containing model, optimizer, epoch, etc.
        filepath (str): Path to save the checkpoint.
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device=None):
    """
    Loads a model checkpoint.

    Args:
        filepath (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (optional): Scheduler to load state into.
        device (torch.device, optional): Device to map the location to.

    Returns:
        int: The epoch number if available in the checkpoint, else 0.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")

    if device is None:
        device = get_device()

    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Assume the checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("epoch", 0)
