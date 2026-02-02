import os
import random
import numpy as np
import torch
import pandas as pd
from copy import deepcopy
from sklearn.metrics import roc_auc_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_roc_auc(y_true, y_pred):
    """
    Calculates the Mean Column-wise ROC AUC score.

    Args:
        y_true: Ground truth labels (N, Num_Classes) - One-hot encoded or binary indicators.
        y_pred: Predicted probabilities (N, Num_Classes).

    Returns:
        float: The macro-averaged ROC AUC score.
    """
    try:
        # average='macro' calculates metrics for each label, and finds their unweighted mean.
        # multi_class='ovr' (One-vs-Rest) is standard for multi-label/multi-class AUC.
        score = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")
    except ValueError:
        # Fallback for edge cases (e.g., only one class present in a small batch)
        score = 0.5
    return score


def compute_class_weights(df, device, load_cached_data=True):
    """
    Computes inverse frequency class weights to handle class imbalance.
    Implements caching mechanism to store/load weights from disk.

    Args:
        df (pd.DataFrame): Training dataframe containing class columns.
        device (torch.device): Device to move the tensor to.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        torch.Tensor: Class weights tensor of shape (Num_Classes,).
    """
    cache_path = os.path.join(Config.WORKING_DIR, "class_weights.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            return torch.tensor(weights, dtype=torch.float32).to(device)
        except Exception:
            # If loading fails, proceed to recompute
            pass

    # 2. Compute from scratch
    classes = Config.CLASSES
    # Filter only the class columns
    if not all(col in df.columns for col in classes):
        raise ValueError(f"Dataframe must contain columns: {classes}")

    # Calculate counts for each class
    # Assuming df contains binary/one-hot columns for the classes
    counts = df[classes].sum().values
    total_samples = len(df)
    num_classes = len(classes)

    # Inverse Frequency Formula: N / (C * count_c)
    # Add epsilon to prevent division by zero
    weights = total_samples / (num_classes * (counts + 1e-6))

    # 3. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    np.save(cache_path, weights)

    return torch.tensor(weights, dtype=torch.float32).to(device)


class ModelEMA:
    """
    Model Exponential Moving Average (EMA).
    Maintains a moving average of model parameters to stabilize training and improve generalization.
    """

    def __init__(self, model, decay=0.999):
        """
        Args:
            model (torch.nn.Module): The model to track.
            decay (float): The decay factor for EMA (default: 0.999).
        """
        self.decay = decay
        self.ema_model = deepcopy(model)
        self.ema_model.eval()

        # Disable gradients for the EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the EMA model parameters based on the current model.

        Args:
            model (torch.nn.Module): The current training model.
        """
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema_model.state_dict()

            for name, param in msd.items():
                if name in esd:
                    ema_param = esd[name]

                    # Update floating point parameters (weights, biases) with EMA
                    if param.dtype.is_floating_point:
                        ema_param.mul_(self.decay).add_(param, alpha=1 - self.decay)
                    else:
                        # Directly copy integer parameters (e.g., num_batches_tracked)
                        ema_param.copy_(param)
