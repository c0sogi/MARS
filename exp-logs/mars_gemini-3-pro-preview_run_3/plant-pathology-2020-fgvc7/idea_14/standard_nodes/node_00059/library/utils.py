import os
import random
import copy
import numpy as np
import torch
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_class_weights(df, load_cached_data=True):
    """
    Calculates inverse frequency class weights to handle class imbalance.
    Implements strict caching logic using .npy files.

    Args:
        df (pd.DataFrame): Training metadata containing class labels.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        torch.Tensor: A tensor of class weights on the configured device.
    """
    # Ensure the working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    cache_path = Config.CLASS_WEIGHTS_PATH

    # 1. Try to load cached data
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights_np = np.load(cache_path)
            return torch.tensor(weights_np, dtype=torch.float32).to(Config.DEVICE)
        except Exception:
            # If loading fails, proceed to compute
            pass

    # 2. Compute data from scratch
    # Calculate counts for each class defined in Config
    class_counts = []
    for label in Config.CLASSES:
        if label in df.columns:
            # Sum the column (works for one-hot or soft labels)
            class_counts.append(df[label].sum())
        else:
            # Fallback if column missing (should not happen based on metadata)
            class_counts.append(0)

    class_counts = np.array(class_counts)
    total_samples = df.shape[0]
    num_classes = len(Config.CLASSES)

    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1.0)

    # Inverse frequency formula: N_total / (N_classes * N_count)
    weights_np = total_samples / (num_classes * class_counts)

    # 3. Save to cache
    np.save(cache_path, weights_np)

    return torch.tensor(weights_np, dtype=torch.float32).to(Config.DEVICE)


class ModelEMA:
    """
    Implements Exponential Moving Average (EMA) for model weights.
    Maintains a shadow copy of the model that is updated smoothly.
    """

    def __init__(self, model, decay=0.9999):
        """
        Args:
            model (torch.nn.Module): The model to track.
            decay (float): The decay factor for EMA updates.
        """
        self.decay = decay
        # Create a deep copy of the model to serve as the shadow model
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.module.to(Config.DEVICE)

    def update(self, model):
        """
        Update the shadow model parameters using the current model parameters.

        Args:
            model (torch.nn.Module): The current training model.
        """
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.module.state_dict()

            for k, v in msd.items():
                if k in esd:
                    # Update floating point parameters (weights, biases, running stats)
                    if v.dtype.is_floating_point:
                        esd[k].copy_(self.decay * esd[k] + (1.0 - self.decay) * v)
                    else:
                        # For integer buffers (e.g., num_batches_tracked), just copy
                        esd[k].copy_(v)
