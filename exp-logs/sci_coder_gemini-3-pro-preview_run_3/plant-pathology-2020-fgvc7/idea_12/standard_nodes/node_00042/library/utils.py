import os
import random
import logging
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


def get_logger(name=__name__):
    """
    Creates and returns a logger with standard formatting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


class ModelEMA:
    """
    Model Exponential Moving Average.
    Maintains a moving average of model parameters for better generalization.
    """

    def __init__(self, model, decay=None, device=None):
        self.decay = decay if decay is not None else Config.model_ema_decay
        self.module = deepcopy(model)
        self.module.eval()
        self.device = device if device is not None else Config.device
        self.module.to(self.device)

    def update(self, model):
        """
        Update the EMA model parameters.
        """
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.module.state_dict().items():
                if k in msd:
                    v.copy_(self.decay * v + (1.0 - self.decay) * msd[k])


def get_class_weights(df, load_cached_data=True):
    """
    Computes or loads inverse frequency class weights.
    Implements strict caching logic using .npy files.
    """
    os.makedirs(Config.working_dir, exist_ok=True)
    cache_path = os.path.join(Config.working_dir, "class_weights.npy")

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights = np.load(cache_path)
            return torch.tensor(weights, dtype=torch.float32).to(Config.device)
        except Exception:
            # If load fails, proceed to recompute
            pass

    # 2. Compute from scratch
    # Ensure df is provided if we are computing
    if df is None:
        raise ValueError(
            "DataFrame is required to compute class weights if cache is missing."
        )

    # Calculate counts for each class label defined in Config
    # Assuming df contains one-hot encoded columns or we can derive them.
    # Based on metadata, columns match Config.class_labels.
    counts = df[Config.class_labels].sum().values

    # Inverse Frequency: N / (C * count)
    # Add small epsilon to avoid division by zero
    total_samples = len(df)
    num_classes = len(Config.class_labels)
    weights = total_samples / (num_classes * (counts + 1e-6))

    # Cast to float32 for torch compatibility
    weights = weights.astype(np.float32)

    # 3. Save to cache
    np.save(cache_path, weights)

    return torch.tensor(weights, dtype=torch.float32).to(Config.device)


def compute_metric(y_true, y_pred):
    """
    Computes the Mean Column-wise ROC AUC.

    Args:
        y_true: Ground truth labels (numpy array), shape (N, num_classes)
        y_pred: Predicted probabilities (numpy array), shape (N, num_classes)

    Returns:
        float: The mean ROC AUC score.
    """
    try:
        # average='macro' computes the metric for each label, and finds their unweighted mean.
        # This is equivalent to mean column-wise ROC AUC.
        score = roc_auc_score(y_true, y_pred, average="macro")
        return score
    except ValueError:
        # Handle cases where a class might not be present in the batch/set
        return 0.5
