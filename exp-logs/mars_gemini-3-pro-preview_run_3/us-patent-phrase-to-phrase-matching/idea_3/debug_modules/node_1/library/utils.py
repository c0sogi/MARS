import os
import re
import random
import numpy as np
import torch
from scipy.stats import pearsonr


def set_seed(seed=42):
    """
    Sets the seed for random number generators in Python, NumPy, and PyTorch
    to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_pearson(predictions, labels):
    """
    Computes the Pearson correlation coefficient between predictions and labels.

    Args:
        predictions (array-like): Predicted scores.
        labels (array-like): Ground truth scores.

    Returns:
        float: The Pearson correlation coefficient.
    """
    # pearsonr returns (statistic, pvalue), we only need the statistic
    return pearsonr(predictions, labels)[0]


def get_optimizer_grouped_parameters(model, learning_rate, weight_decay, llrd_decay):
    """
    Groups model parameters for the optimizer, applying Layer-wise Learning Rate Decay (LLRD)
    and handling weight decay exclusion for bias/LayerNorm terms.

    Args:
        model (torch.nn.Module): The model to optimize.
        learning_rate (float): The base learning rate (applied to the head).
        weight_decay (float): The weight decay coefficient.
        llrd_decay (float): The multiplicative decay factor for lower layers.

    Returns:
        list: A list of dictionaries defining parameter groups for the optimizer.
    """
    # 1. Identify the maximum layer index in the backbone
    # DeBERTa/RoBERTa usually have names like '...encoder.layer.0...', '...encoder.layer.23...'
    max_layer_index = 0
    for name, _ in model.named_parameters():
        match = re.search(r"layer\.(\d+)", name)
        if match:
            layer_idx = int(match.group(1))
            if layer_idx > max_layer_index:
                max_layer_index = layer_idx

    # Define depth indices
    # Embeddings -> 0
    # Layer 0 -> 1
    # ...
    # Layer N -> N + 1
    # Head -> N + 2
    head_depth = max_layer_index + 2

    # 2. Group parameters
    # Key: (weight_decay_value, calculated_learning_rate)
    # Value: list of parameters
    param_groups = {}

    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine Weight Decay
        # If parameter name contains any no_decay string, set wd to 0
        if any(nd in name for nd in no_decay):
            wd = 0.0
        else:
            wd = weight_decay

        # Determine Depth and Learning Rate
        if "embeddings" in name:
            depth = 0
        elif "layer." in name:
            match = re.search(r"layer\.(\d+)", name)
            if match:
                depth = int(match.group(1)) + 1
            else:
                # Fallback if regex fails but 'layer.' is present (unlikely)
                depth = head_depth
        else:
            # Parameters not in embeddings or encoder layers are considered 'Head'
            # This includes the pooler, final layernorm, and the custom regression head
            depth = head_depth

        # Calculate LLRD
        # LR = base_lr * (decay ^ (max_depth - depth))
        # Head (depth=max) -> decay^0 = 1.0
        # Embeddings (depth=0) -> decay^max
        scale = llrd_decay ** (head_depth - depth)
        lr = learning_rate * scale

        # Add to group
        group_key = (wd, lr)
        if group_key not in param_groups:
            param_groups[group_key] = []
        param_groups[group_key].append(param)

    # 3. Format for Optimizer
    optimizer_grouped_parameters = []
    for (wd, lr), params in param_groups.items():
        optimizer_grouped_parameters.append(
            {"params": params, "weight_decay": wd, "lr": lr}
        )

    return optimizer_grouped_parameters
