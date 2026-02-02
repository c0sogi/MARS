import os
import random
import numpy as np
import torch
from scipy import stats


def seed_everything(seed: int = 42):
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


def compute_score(y_true, y_pred):
    """
    Computes the Pearson correlation coefficient between true and predicted scores.
    Handles both numpy arrays and torch tensors.

    Args:
        y_true: Array-like or Tensor of ground truth scores.
        y_pred: Array-like or Tensor of predicted scores.

    Returns:
        float: The Pearson correlation coefficient.
    """
    # Convert torch tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are flattened 1D arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Compute Pearson correlation
    # stats.pearsonr returns (statistic, p-value), we need the statistic
    score, _ = stats.pearsonr(y_true, y_pred)
    return score


def get_llrd_optimizer_params(model, learning_rate, weight_decay=0.01, llrd_decay=0.9):
    """
    Constructs the optimizer parameter groups with Layer-wise Learning Rate Decay (LLRD).

    Strategy:
    - The task head (classifier) gets the full `learning_rate`.
    - The top encoder layer gets `learning_rate * llrd_decay`.
    - Each subsequent lower layer gets an additional factor of `llrd_decay`.
    - Embeddings get the lowest learning rate.
    - Weight decay is applied to weights but excluded for biases and LayerNorms.

    Args:
        model: The HuggingFace model (e.g., DeBERTa).
        learning_rate: The base learning rate (max LR applied to head).
        weight_decay: The weight decay coefficient.
        llrd_decay: The multiplicative decay factor per layer (e.g., 0.9).

    Returns:
        list: A list of dictionaries suitable for the optimizer.
    """

    # Determine the number of layers (DeBERTa-v3-large typically has 24)
    if hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    else:
        num_layers = 24  # Fallback default

    named_parameters = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # Helper to calculate the learning rate multiplier based on parameter name
    def get_lr_multiplier(name):
        # 1. Task Head (Classifier/Pooler)
        # Parameters not belonging to the backbone (e.g., 'deberta') are the head.
        # Adjust 'deberta' if using a different backbone prefix.
        if "deberta" not in name:
            return 1.0

        # 2. Embeddings
        # Lowest layer, gets the most decay
        if "embeddings" in name:
            return llrd_decay ** (num_layers + 1)

        # 3. Encoder Layers
        # Format usually: deberta.encoder.layer.{i}.xxx
        if "encoder.layer" in name:
            try:
                parts = name.split(".")
                # Find the index following 'layer'
                layer_idx_loc = parts.index("layer")
                layer_index = int(parts[layer_idx_loc + 1])

                # Top layer (index = num_layers - 1) -> decay^1
                # Bottom layer (index = 0) -> decay^(num_layers)
                return llrd_decay ** (num_layers - layer_index)
            except (ValueError, IndexError):
                # Fallback if parsing fails
                return llrd_decay ** (num_layers + 1)

        # 4. Other Encoder components (e.g., final LayerNorm after encoder)
        if "encoder.LayerNorm" in name:
            # Treat similar to the top layer
            return llrd_decay**1

        # Default fallback
        return llrd_decay ** (num_layers + 1)

    # Group parameters by (learning_rate, weight_decay)
    groups = {}  # Key: (lr_value, wd_value) -> list of params

    for name, param in named_parameters:
        if not param.requires_grad:
            continue

        # Calculate LR
        multiplier = get_lr_multiplier(name)
        lr = learning_rate * multiplier

        # Calculate Weight Decay
        if any(nd in name for nd in no_decay):
            wd = 0.0
        else:
            wd = weight_decay

        key = (lr, wd)
        if key not in groups:
            groups[key] = []
        groups[key].append(param)

    # Construct the final list
    optimizer_grouped_parameters = []
    for (lr, wd), params in groups.items():
        optimizer_grouped_parameters.append(
            {"params": params, "weight_decay": wd, "lr": lr}
        )

    return optimizer_grouped_parameters
