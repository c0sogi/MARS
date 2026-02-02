import os
import random
import re
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_qwk(y_true, y_pred):
    """
    Computes the Quadratic Weighted Kappa (QWK) score.

    Args:
        y_true (array-like): Ground truth scores (integers).
        y_pred (array-like): Predicted scores (continuous or integers).

    Returns:
        float: The QWK score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Clip predictions to valid range [1, 6] and round to nearest integer
    # This is necessary because the model outputs continuous regression values
    y_pred_clipped = np.clip(y_pred, 1, 6)
    y_pred_rounded = np.round(y_pred_clipped).astype(int)
    y_true = y_true.astype(int)

    return cohen_kappa_score(y_true, y_pred_rounded, weights="quadratic")


def get_llrd_optimizer_params(model, base_lr, head_lr, weight_decay, llrd_decay):
    """
    Groups model parameters for Layer-wise Learning Rate Decay (LLRD).

    This function assigns different learning rates to different layers of the backbone:
    - The Head gets `head_lr`.
    - The top transformer layer gets `base_lr`.
    - Lower layers get exponentially decaying learning rates: base_lr * (decay ^ depth).
    - Embeddings get the lowest learning rate.

    Args:
        model (torch.nn.Module): The model to optimize.
        base_lr (float): Learning rate for the top layer of the backbone.
        head_lr (float): Learning rate for the regression head.
        weight_decay (float): Weight decay coefficient.
        llrd_decay (float): Decay factor for LLRD (e.g., 0.9).

    Returns:
        list: A list of dictionaries suitable for torch.optim.Optimizer.
    """
    # Define parameters to exclude from weight decay
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # Detect the number of layers in the backbone automatically
    # We look for patterns like "encoder.layer.23." to find the max index
    max_layer_index = 0
    for name, _ in model.named_parameters():
        match = re.search(r"encoder\.layer\.(\d+)\.", name)
        if match:
            layer_idx = int(match.group(1))
            if layer_idx > max_layer_index:
                max_layer_index = layer_idx

    num_layers = max_layer_index + 1

    # Initialize groups
    # We use a dictionary to map (lr, weight_decay) tuples to parameter lists
    # This avoids creating hundreds of individual parameter groups
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine Weight Decay
        if any(nd in name for nd in no_decay):
            wd = 0.0
        else:
            wd = weight_decay

        # Determine Learning Rate
        if "embeddings" in name:
            # Embeddings get the strongest decay (furthest from output)
            lr = base_lr * (llrd_decay ** (num_layers + 1))
        elif "encoder.layer" in name:
            # Extract layer index
            match = re.search(r"encoder\.layer\.(\d+)\.", name)
            if match:
                layer_idx = int(match.group(1))
                # Layer N-1 (top) gets decay^0, Layer 0 gets decay^(N-1)
                lr = base_lr * (llrd_decay ** (num_layers - 1 - layer_idx))
            else:
                # Fallback if regex fails but name contains encoder.layer (unlikely)
                lr = base_lr
        else:
            # Head parameters (fc, classifier, pooler, etc.)
            lr = head_lr

        # Add to group
        group_key = (lr, wd)
        if group_key not in param_groups:
            param_groups[group_key] = []
        param_groups[group_key].append(param)

    # Convert to list format expected by Optimizer
    optimizer_grouped_parameters = []
    for (lr, wd), params in param_groups.items():
        optimizer_grouped_parameters.append(
            {"params": params, "lr": lr, "weight_decay": wd}
        )

    return optimizer_grouped_parameters
