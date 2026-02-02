import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def laplace_log_likelihood(y_true, y_pred, sigma):
    """
    Computes the modified Laplace Log Likelihood metric.

    Formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Args:
        y_true (np.array or torch.Tensor): Ground truth FVC values.
        y_pred (np.array or torch.Tensor): Predicted FVC values.
        sigma (np.array or torch.Tensor): Predicted confidence (standard deviation).

    Returns:
        float: The average metric score (higher is better, values are negative).
    """
    # Detach and move to CPU if inputs are tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().numpy()

    # Ensure float precision
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    sigma = sigma.astype(np.float64)

    # Constants
    MAX_ERROR = 1000.0
    SIGMA_CLIP = 70.0
    SQRT_2 = np.sqrt(2)

    # Clip sigma
    sigma_clipped = np.maximum(sigma, SIGMA_CLIP)

    # Calculate absolute error and clip it
    abs_error = np.abs(y_true - y_pred)
    delta = np.minimum(abs_error, MAX_ERROR)

    # Compute metric
    metric = -(SQRT_2 * delta) / sigma_clipped - np.log(SQRT_2 * sigma_clipped)

    return np.mean(metric)


def average_weights(model, checkpoint_paths):
    """
    Performs Stochastic Weight Averaging (SWA) by averaging the weights
    from a list of model checkpoints.

    Args:
        model (torch.nn.Module): The model instance to load weights into.
        checkpoint_paths (list): List of file paths to the saved checkpoints.

    Returns:
        torch.nn.Module: The model with averaged weights.
    """
    if not checkpoint_paths:
        return model

    # Helper to extract state_dict if wrapped
    def load_state_dict_safe(path):
        data = torch.load(path, map_location=Config.DEVICE)
        if isinstance(data, dict):
            if "state_dict" in data:
                return data["state_dict"]
            elif "model_state_dict" in data:
                return data["model_state_dict"]
        return data

    # Initialize sum with the first checkpoint
    avg_state_dict = load_state_dict_safe(checkpoint_paths[0])

    # Accumulate weights from remaining checkpoints
    for path in checkpoint_paths[1:]:
        state_dict = load_state_dict_safe(path)
        for key in avg_state_dict:
            avg_state_dict[key] += state_dict[key]

    # Average the weights
    n = len(checkpoint_paths)
    for key in avg_state_dict:
        avg_state_dict[key] = avg_state_dict[key] / n

    # Load the averaged weights into the model
    model.load_state_dict(avg_state_dict)

    return model
