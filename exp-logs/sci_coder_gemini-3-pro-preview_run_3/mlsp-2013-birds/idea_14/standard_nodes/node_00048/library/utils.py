import os
import random
import numpy as np
import torch
import shutil
from copy import deepcopy
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModelEMA:
    """
    Model Exponential Moving Average (EMA).
    Maintains a shadow copy of the model that is updated using an exponential decay.
    Useful for stabilizing training on small datasets.
    """

    def __init__(self, model, decay: float = 0.95, device=None):
        """
        Args:
            model: The source model to track.
            decay: The decay rate for the moving average.
                   Lower values (e.g. 0.95) allow faster adaptation for short training schedules.
            device: The device to store the shadow model on.
        """
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device:
            self.module.to(self.device)

    def update(self, model):
        """
        Update the shadow model parameters based on the current model parameters.
        Args:
            model: The current training model.
        """
        with torch.no_grad():
            # Iterate over the state dictionary to handle both parameters and buffers (e.g. BN stats)
            msd = model.state_dict()
            esd = self.module.state_dict()

            for k in msd:
                model_v = msd[k].detach()
                ema_v = esd[k]

                if self.device:
                    model_v = model_v.to(self.device)

                # Update logic: shadow = decay * shadow + (1 - decay) * new
                # We use copy_ to update in-place
                esd[k].copy_(ema_v * self.decay + model_v * (1.0 - self.decay))


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the mean ROC AUC score, robust to missing classes in the input batch.

    Args:
        y_true: Ground truth labels (N, NumClasses), numpy array.
        y_pred: Predicted probabilities (N, NumClasses), numpy array.

    Returns:
        float: The mean ROC AUC over valid classes.
    """
    n_classes = y_true.shape[1]
    aucs = []

    for i in range(n_classes):
        # Check if the class has both positive and negative samples
        # roc_auc_score throws an error if only one class is present in y_true
        if len(np.unique(y_true[:, i])) > 1:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(auc)
            except ValueError:
                # Fallback for edge cases not caught by unique check
                continue

    if not aucs:
        return 0.0

    return np.mean(aucs)


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    """
    Saves the model checkpoint.

    Args:
        state: Dict containing model state, optimizer state, etc.
        is_best: Boolean indicating if this is the best model so far.
        filename: Base filename for the checkpoint.
    """
    filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(Config.CHECKPOINT_DIR, f"best_{filename}")
        shutil.copyfile(filepath, best_path)


def load_checkpoint(filename, model, optimizer=None, device="cpu"):
    """
    Loads a checkpoint.

    Args:
        filename: Filename of the checkpoint to load (relative to CHECKPOINT_DIR or absolute).
        model: The model to load weights into.
        optimizer: (Optional) Optimizer to load state into.
        device: Device to map the storage to.

    Returns:
        start_epoch, best_score
    """
    if not os.path.isabs(filename):
        filepath = os.path.join(Config.CHECKPOINT_DIR, filename)
    else:
        filepath = filename

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["state_dict"])

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint.get("epoch", 0)
    best_score = checkpoint.get("best_score", 0.0)

    return start_epoch, best_score
