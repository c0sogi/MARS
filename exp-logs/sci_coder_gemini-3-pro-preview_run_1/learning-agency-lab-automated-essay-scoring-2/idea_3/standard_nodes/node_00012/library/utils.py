import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score


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


def compute_qwk(y_true, y_pred):
    """
    Computes the Quadratic Weighted Kappa metric.

    Args:
        y_true: Array-like of true integer scores.
        y_pred: Array-like of predicted integer scores.

    Returns:
        float: The QWK score.
    """
    # Ensure inputs are numpy arrays and integer type
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def get_optimizer_params(
    model, encoder_lr, decoder_lr, weight_decay=0.01, llrd_decay=0.9
):
    """
    Constructs optimizer parameter groups with Layer-wise Learning Rate Decay (LLRD).

    This function groups model parameters to assign specific learning rates and weight decay settings.
    - The 'head' (classifier) gets `decoder_lr`.
    - The transformer backbone layers get learning rates that decay as the depth increases
      (lower layers have lower learning rates).
    - Bias and LayerNorm parameters are excluded from weight decay.

    Args:
        model: The PyTorch model (nn.Module).
        encoder_lr: Base learning rate for the top transformer layer.
        decoder_lr: Learning rate for the classifier head.
        weight_decay: Weight decay coefficient for regularization.
        llrd_decay: Multiplicative decay factor for lower layers (0 < llrd_decay <= 1).

    Returns:
        list: A list of dictionaries suitable for the optimizer (e.g., AdamW).
    """
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # 1. Identify the number of layers in the backbone
    # We scan parameter names for the pattern 'layer.{i}.' typical in DeBERTa/BERT models
    layer_indices = set()
    for name, _ in model.named_parameters():
        if "layer." in name:
            parts = name.split(".")
            try:
                # Find the integer index following "layer"
                idx = int(parts[parts.index("layer") + 1])
                layer_indices.add(idx)
            except (ValueError, IndexError):
                continue

    # If layers found, max index + 1 is the count. Otherwise assume 0 (e.g., linear model).
    num_layers = max(layer_indices) + 1 if layer_indices else 0

    # 2. Group parameters
    # We use a dictionary to aggregate parameters that share the same (lr, weight_decay)
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # --- Determine Learning Rate ---
        if "embeddings" in name:
            # Embeddings are at the very bottom, so they get the most decay
            # Depth = num_layers (conceptually below layer 0)
            lr = encoder_lr * (llrd_decay**num_layers)

        elif "layer." in name:
            # Transformer Encoder Layers
            try:
                parts = name.split(".")
                layer_idx = int(parts[parts.index("layer") + 1])

                # Calculate distance from the top layer.
                # Top layer (index = num_layers - 1) gets encoder_lr.
                # Layer below gets encoder_lr * llrd_decay, etc.
                distance = (num_layers - 1) - layer_idx
                lr = encoder_lr * (llrd_decay**distance)
            except (ValueError, IndexError):
                # Fallback if parsing fails
                lr = encoder_lr

        else:
            # Head / Classifier parameters (not embeddings, not encoder layers)
            # These get the decoder_lr (usually higher than encoder_lr)
            lr = decoder_lr

        # --- Determine Weight Decay ---
        if any(nd in name for nd in no_decay):
            wd = 0.0
        else:
            wd = weight_decay

        # --- Add to Group ---
        key = (lr, wd)
        if key not in param_groups:
            param_groups[key] = []
        param_groups[key].append(param)

    # 3. Format for Optimizer
    optimizer_grouped_parameters = []
    for (lr, wd), params in param_groups.items():
        optimizer_grouped_parameters.append(
            {"params": params, "lr": lr, "weight_decay": wd}
        )

    return optimizer_grouped_parameters
