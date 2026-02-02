import os
import copy
import torch
import numpy as np
from library.config import Config


class ModelEMA:
    """
    Model Exponential Moving Average (EMA).
    Maintains a shadow copy of the model weights that is updated using an exponential moving average.
    This technique helps in stabilizing training and often leads to better generalization.
    """

    def __init__(self, model, decay=Config.EMA_DECAY, device=Config.DEVICE):
        """
        Args:
            model (torch.nn.Module): The model to track.
            decay (float): The decay factor for EMA (default: Config.EMA_DECAY).
            device (str): The device to store the EMA model on (default: Config.DEVICE).
        """
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.model.to(device)

        # Disable gradients for the EMA model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the EMA model parameters based on the current model parameters.

        Args:
            model (torch.nn.Module): The current training model.
        """
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.model.state_dict()

            for name, param in msd.items():
                if name in esd:
                    if param.dtype.is_floating_point:
                        # shadow = decay * shadow + (1 - decay) * new
                        # equivalent to: shadow.mul_(decay).add_(new, alpha=1-decay)
                        esd[name].mul_(self.decay).add_(param, alpha=1.0 - self.decay)
                    else:
                        # Directly copy non-floating point parameters (e.g., integer buffers)
                        esd[name].copy_(param)

    def get_model(self):
        """Returns the stored EMA model."""
        return self.model


class EarlyStopping:
    """
    Early stops the training if the monitored validation metric doesn't improve after a given patience.
    """

    def __init__(
        self,
        patience=Config.EARLY_STOPPING_PATIENCE,
        delta=0.0,
        mode="max",
        verbose=True,
    ):
        """
        Args:
            patience (int): How long to wait after the last time the validation metric improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): One of {'min', 'max'}. 'max' for metrics like AUC, 'min' for Loss.
            verbose (bool): If True, prints a message for each validation improvement.
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        # Initialize best score based on mode
        self.val_score_best = -np.inf if mode == "max" else np.inf

        if mode == "min":
            self.monitor_op = np.less
            # For min, we want score < best - delta
            self.delta_op = lambda x, d: x - d
        elif mode == "max":
            self.monitor_op = np.greater
            # For max, we want score > best + delta
            self.delta_op = lambda x, d: x + d
        else:
            raise ValueError(f"EarlyStopping mode must be 'min' or 'max', got {mode}")

    def __call__(self, score, model, model_path):
        """
        Args:
            score (float): The metric value to monitor.
            model (torch.nn.Module or ModelEMA): The model to save.
            model_path (str): Path to save the checkpoint.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, model_path)
        elif self.monitor_op(score, self.delta_op(self.best_score, self.delta)):
            self.best_score = score
            self.save_checkpoint(score, model, model_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, model, model_path):
        """Saves model when validation metric improves."""
        if self.verbose:
            print(
                f"Validation metric improved ({self.val_score_best} --> {score}). Saving model to {model_path}"
            )

        self.val_score_best = score

        # Handle ModelEMA wrapper by extracting the underlying model
        if isinstance(model, ModelEMA):
            torch.save(model.get_model().state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)


def apply_mixup(x, y, alpha=Config.MIXUP_ALPHA, device=Config.DEVICE):
    """
    Applies Mixup regularization to the batch.

    Args:
        x (torch.Tensor): Input batch of images (B, C, H, W).
        y (torch.Tensor): Input batch of labels (B, NumClasses).
        alpha (float): Mixup alpha parameter.
        device (str): Device to perform operations on.

    Returns:
        mixed_x (torch.Tensor): Mixed images.
        y_a (torch.Tensor): Labels for the first component.
        y_b (torch.Tensor): Labels for the second component.
        lam (float): Lambda mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam
