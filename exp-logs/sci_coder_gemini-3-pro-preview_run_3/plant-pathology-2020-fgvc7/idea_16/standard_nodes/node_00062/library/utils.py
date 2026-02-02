import os
import random
import numpy as np
import torch
import pandas as pd
from copy import deepcopy
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across various libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(df, load_cached_data=True):
    """
    Computes inverse frequency class weights for the loss function.
    Implements caching to ./working/idea_16/class_weights.npy
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = os.path.join(Config.WORKING_DIR, "class_weights.npy")

    # 1. Try to load from cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            return torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)
        except Exception:
            # If loading fails, proceed to compute
            pass

    # 2. Compute from scratch
    # Identify the target labels
    if "stratify_label" in df.columns:
        # Use the pre-computed label column if available
        y_labels = df["stratify_label"].values
    else:
        # Fallback: derive from one-hot columns based on Config.LABELS
        y_labels = df[Config.LABELS].idxmax(axis=1).values

    # Map string labels to indices
    label_to_idx = {label: i for i, label in enumerate(Config.LABELS)}
    y_indices = np.array([label_to_idx[label] for label in y_labels])

    # Compute balanced weights using sklearn
    classes = np.arange(len(Config.LABELS))
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_indices
    )

    # 3. Save to cache
    np.save(cache_path, weights)

    return torch.tensor(weights, dtype=torch.float32).to(Config.DEVICE)


class ModelEMA:
    """
    Model Exponential Moving Average.
    Maintains a moving average of model parameters for better generalization.
    """

    def __init__(self, model, decay=0.999, device=None):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device:
            self.module.to(device)

    def update(self, model):
        """
        Update the shadow parameters using the current model parameters.
        shadow = decay * shadow + (1 - decay) * current
        """
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.module.state_dict()
            for k in msd:
                if msd[k].dtype.is_floating_point:
                    esd[k].copy_(self.decay * esd[k] + (1.0 - self.decay) * msd[k])


def calculate_metric(y_true, y_pred):
    """
    Computes the Mean Column-wise ROC AUC.

    Args:
        y_true: Ground truth labels (N, Num_Classes), can be numpy or tensor (one-hot).
        y_pred: Predicted probabilities (N, Num_Classes), can be numpy or tensor.

    Returns:
        float: The mean ROC AUC score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate ROC AUC
    # 'average="macro"' computes the metric for each label, and finds their unweighted mean.
    # This is equivalent to Mean Column-wise ROC AUC for multi-label/one-hot inputs.
    try:
        score = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        # Handle edge cases where a class might be missing in the batch (e.g., during debugging)
        score = 0.0

    return score
