import os
import random
import re
import numpy as np
import torch
from scipy import stats
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metric(y_true, y_pred):
    """
    Computes the Mean Column-wise Spearman's Correlation Coefficient.

    Args:
        y_true: numpy array or torch tensor of shape (N, 30) containing true labels.
        y_pred: numpy array or torch tensor of shape (N, 30) containing predicted probabilities.

    Returns:
        float: The mean Spearman's correlation coefficient.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    scores = []
    num_targets = y_true.shape[1]

    for i in range(num_targets):
        # Extract columns
        t = y_true[:, i]
        p = y_pred[:, i]

        # Handle constant values which cause NaN in correlation
        # If variance is effectively 0, correlation is undefined.
        if np.std(t) < 1e-9 or np.std(p) < 1e-9:
            scores.append(0.0)
        else:
            score = stats.spearmanr(t, p).correlation
            if np.isnan(score):
                scores.append(0.0)
            else:
                scores.append(score)

    return np.mean(scores)


def get_optimizer_params(model, encoder_lr, head_lr, weight_decay):
    """
    Constructs the parameter groups for the optimizer with simple Differential Learning Rates
    and specific weight decay handling.

    Args:
        model: The PyTorch model.
        encoder_lr: Learning rate for the backbone.
        head_lr: Learning rate for the head.
        weight_decay: Weight decay coefficient.

    Returns:
        list: A list of dictionaries defining parameter groups compatible with torch.optim.Optimizer.
    """
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # --- Determine Learning Rate ---
        lr = head_lr  # Default to head LR

        # Check if the parameter belongs to the backbone (Transformer)
        is_backbone = False
        if any(
            k in name
            for k in [
                "roberta",
                "bert",
                "distilbert",
                "backbone",
                "encoder",
                "embeddings",
            ]
        ):
            # Exclude explicit head names if they accidentally matched
            if not any(k in name for k in ["head", "classifier", "fc", "linear"]):
                is_backbone = True
        elif "layer." in name:
            is_backbone = True

        if is_backbone:
            lr = encoder_lr
        else:
            lr = head_lr

        # --- Determine Weight Decay ---
        # Exclude bias and LayerNorm parameters from weight decay
        if any(x in name for x in ["bias", "LayerNorm", "norm", "ln_"]):
            wd = 0.0
        else:
            wd = weight_decay

        # --- Grouping ---
        # Group parameters by unique (lr, weight_decay) pairs to minimize number of groups
        group_key = (lr, wd)
        if group_key not in param_groups:
            param_groups[group_key] = []
        param_groups[group_key].append(param)

    # Convert to list of dictionaries
    optimizer_grouped_parameters = []
    for (lr, wd), params in param_groups.items():
        optimizer_grouped_parameters.append(
            {"params": params, "lr": lr, "weight_decay": wd}
        )

    return optimizer_grouped_parameters
