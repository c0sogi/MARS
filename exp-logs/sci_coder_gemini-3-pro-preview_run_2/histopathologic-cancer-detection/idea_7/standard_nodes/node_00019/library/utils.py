import os
import random
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from sklearn.metrics import roc_auc_score
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class ModelEMA:
    """
    Exponential Moving Average for model parameters.
    Maintains a shadow model that updates as:
    shadow = decay * shadow + (1 - decay) * model
    """

    def __init__(self, model, decay=0.9999, device=None):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device:
            self.module.to(device)

    def update(self, model):
        """
        Update the shadow model weights.
        """
        with torch.no_grad():
            # Handle potential DataParallel wrapping
            if hasattr(model, "module"):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()

            ema_state = self.module.state_dict()

            for name, param in model_state.items():
                if name in ema_state:
                    if param.dtype.is_floating_point:
                        ema_state[name].copy_(
                            self.decay * ema_state[name] + (1.0 - self.decay) * param
                        )
                    else:
                        ema_state[name].copy_(param)


class Mixup:
    """
    Implements Mixup regularization.
    """

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x, y):
        """
        Applies Mixup to the batch x and targets y.

        Args:
            x (torch.Tensor): Input batch of images.
            y (torch.Tensor): Input batch of labels.

        Returns:
            mixed_x (torch.Tensor): Mixed images.
            y_a (torch.Tensor): Original labels.
            y_b (torch.Tensor): Shuffled labels.
            lam (float): Mixing coefficient.
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam


class MetricMonitor:
    """
    Tracks training and validation metrics (Loss and AUC).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val_loss_sum = 0
        self.val_loss_count = 0
        self.predictions = []
        self.targets = []

    def update(self, loss, batch_size):
        """
        Updates the running average of the loss.
        """
        self.val_loss_sum += loss * batch_size
        self.val_loss_count += batch_size

    def update_predictions(self, preds, targets):
        """
        Stores predictions and targets for global AUC calculation.

        Args:
            preds (torch.Tensor or np.array): Predicted probabilities (sigmoid output).
            targets (torch.Tensor or np.array): Ground truth labels.
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self.predictions.append(preds)
        self.targets.append(targets)

    def get_avg_loss(self):
        """
        Returns the average loss over the accumulated batches.
        """
        if self.val_loss_count == 0:
            return 0.0
        return self.val_loss_sum / self.val_loss_count

    def get_auc(self):
        """
        Calculates the ROC AUC score using all accumulated predictions.
        """
        if not self.predictions or not self.targets:
            return 0.0

        all_preds = np.concatenate(self.predictions)
        all_targets = np.concatenate(self.targets)

        # Handle edge case where only one class is present in the batch/epoch
        if len(np.unique(all_targets)) < 2:
            return 0.5

        return roc_auc_score(all_targets, all_preds)


def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state (dict): State dictionary containing model weights, optimizer state, etc.
        is_best (bool): Whether this checkpoint represents the best model so far.
        checkpoint_dir (str): Directory to save the file.
        filename (str): Name of the checkpoint file (e.g., 'checkpoint_fold_0.pth').
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        # Infer best model filename from the checkpoint filename
        # e.g., 'checkpoint_fold_0.pth' -> 'best_model_fold_0.pth'
        if "checkpoint" in filename:
            best_filename = filename.replace("checkpoint", "best_model")
        else:
            best_filename = "best_" + filename

        best_filepath = os.path.join(checkpoint_dir, best_filename)
        torch.save(state, best_filepath)
